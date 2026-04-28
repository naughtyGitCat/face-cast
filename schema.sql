-- face-cluster SQLite schema
-- 设计目标:
--   1. 多模型友好: 同一张脸可有多个模型的 embedding 并存 (历史快照)
--   2. 缓存 face crop: 升级模型时不需重新解码视频
--   3. 聚类历史: 每次 HDBSCAN 跑一次记一笔, 可对比
--   4. NFO 持久化无关: 文件层面 NFO 由 client 改写, 此库只是工作 DB

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- ─────────────────────────────────────────────────────────────────────────
-- 1. frames: 视频中被采样的帧 (帧时间戳 + 来源视频, 唯一标识一帧位置)
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path  TEXT    NOT NULL,
    frame_ms    INTEGER NOT NULL,         -- 视频内时间戳 (毫秒)
    width       INTEGER,
    height      INTEGER,
    sha1        TEXT,                     -- 帧像素 hash (可选, 去重用)
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (video_path, frame_ms)
);

CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_path);

-- ─────────────────────────────────────────────────────────────────────────
-- 2. faces: 帧上检测到的人脸 (跟检测器版本相关, 但 bbox 在视觉空间稳定)
--    crop_jpeg 缓存裁剪好的脸图 (224×224, 多 20% padding) → 升级模型不用回视频
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    bbox_x1     INTEGER NOT NULL,
    bbox_y1     INTEGER NOT NULL,
    bbox_x2     INTEGER NOT NULL,
    bbox_y2     INTEGER NOT NULL,
    det_score   REAL,
    detector    TEXT NOT NULL,            -- 'retinaface_buffalo_l' / 'scrfd_2024' ...
    age         INTEGER,
    sex         INTEGER,                  -- 0=female, 1=male (InsightFace 约定)
    crop_jpeg   BLOB,                     -- ~10-30 KB, 升级模型时直接喂这个
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_faces_frame ON faces(frame_id);

-- ─────────────────────────────────────────────────────────────────────────
-- 3. embeddings: 一张脸 × 一个模型 = 一行向量
--    同一张脸可以有多个模型的 embedding (升级模型时新增, 老的保留)
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id         INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    model_name      TEXT NOT NULL,        -- 'buffalo_l'
    model_version   TEXT NOT NULL,        -- '2024.01.15-onnx'
    dim             INTEGER NOT NULL,     -- 512
    vector          BLOB NOT NULL,        -- float32 × dim, little-endian
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (face_id, model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model
    ON embeddings(model_name, model_version);

-- ─────────────────────────────────────────────────────────────────────────
-- 4. cluster_runs: 一次 HDBSCAN 跑完是一个 run
--    可以对比不同模型 / 不同参数的聚类结果
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cluster_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    algo            TEXT NOT NULL,        -- 'hdbscan'
    params_json     TEXT,                 -- '{"min_cluster_size":3,"metric":"cosine"}'
    n_embeddings    INTEGER,              -- 参与本次聚类的 embedding 数
    n_clusters      INTEGER,              -- 输出 cluster 数 (不含噪声)
    n_noise         INTEGER,              -- 被标 -1 的样本数
    is_active       INTEGER DEFAULT 0,    -- 当前生效的 run? 1=是
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_active ON cluster_runs(is_active);

-- ─────────────────────────────────────────────────────────────────────────
-- 5. clusters: 每个 run 内的具体 cluster (匿名 ID, 用户后期可命名)
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clusters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES cluster_runs(id) ON DELETE CASCADE,
    cluster_idx     INTEGER NOT NULL,    -- HDBSCAN 输出: 0,1,2,... -1=噪声
    size            INTEGER,             -- 该 cluster 的人脸数
    human_name      TEXT,                -- 用户后期填: '演员A' / '李雅' / NULL
    centroid_blob   BLOB,                -- 该 cluster embedding 的均值 (加速增量识别)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, cluster_idx)
);

CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_clusters_named ON clusters(human_name);

-- ─────────────────────────────────────────────────────────────────────────
-- 6. face_cluster: 人脸 ↔ cluster 多对多 (一张脸在不同 run 里属于不同 cluster)
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS face_cluster (
    face_id     INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    cluster_id  INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
    PRIMARY KEY (face_id, cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_fc_cluster ON face_cluster(cluster_id);

-- ─────────────────────────────────────────────────────────────────────────
-- 7. nfo_state: 给每个 video 记一行, 知道 NFO 写过什么 cluster, 后期增量改写
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nfo_state (
    video_path      TEXT PRIMARY KEY,
    nfo_path        TEXT,                 -- 实际 NFO 文件路径
    actors_json     TEXT,                 -- 当前 NFO 里写过哪些 actor (JSON 数组)
    last_run_id     INTEGER REFERENCES cluster_runs(id),
    last_written_at TIMESTAMP
);

-- ─────────────────────────────────────────────────────────────────────────
-- 8. work_log: 每次 phase2 跑了什么, 进度/失败排查
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS work_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    op          TEXT NOT NULL,            -- 'extract' / 'embed' / 'cluster' / 'nfo'
    target      TEXT,                     -- video_path / face_id / run_id 等
    status      TEXT NOT NULL,            -- 'ok' / 'skip' / 'fail'
    detail      TEXT,                     -- 失败原因 / 跳过原因
    duration_ms INTEGER,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_log_op ON work_log(op, status);
