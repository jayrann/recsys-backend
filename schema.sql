-- ═══════════════════════════════════════════════════════════════════════════
-- schema.sql
-- MySQL database schema for the Multi-Criteria Recommender System (MCRS)
-- Run: mysql -u root -p < schema.sql
-- ═══════════════════════════════════════════════════════════════════════════

CREATE DATABASE IF NOT EXISTS mcrs_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE mcrs_db;

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 1: Users
-- Stores all registered user accounts and their roles.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Users (
    user_id       INT          NOT NULL AUTO_INCREMENT,
    username      VARCHAR(50)  NOT NULL UNIQUE,
    email         VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role          ENUM('user', 'admin') NOT NULL DEFAULT 'user',
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login    DATETIME     NULL,

    PRIMARY KEY (user_id),
    INDEX idx_users_username (username),
    INDEX idx_users_email    (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 2: Movies
-- Stores the film catalogue metadata.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Movies (
    movie_id     INT          NOT NULL,
    title        VARCHAR(255) NOT NULL,
    release_year SMALLINT     NULL,
    genres       VARCHAR(255) NULL,
    synopsis     TEXT         NULL,
    poster_url   VARCHAR(500) NULL,

    PRIMARY KEY (movie_id),
    INDEX idx_movies_title (title(50))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 3: Ratings
-- Stores all user-submitted multi-criteria ratings.
-- Each (user_id, movie_id) pair is unique — one rating per user per movie.
-- If a user re-rates a movie, the existing record is updated (ON DUPLICATE KEY).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Ratings (
    user_id          INT            NOT NULL,
    movie_id         INT            NOT NULL,
    storyline        DECIMAL(4, 2)  NOT NULL CHECK (storyline BETWEEN 1 AND 5),
    acting           DECIMAL(4, 2)  NOT NULL CHECK (acting    BETWEEN 1 AND 5),
    visuals          DECIMAL(4, 2)  NOT NULL CHECK (visuals   BETWEEN 1 AND 5),
    emotional_impact DECIMAL(4, 2)  NOT NULL CHECK (emotional_impact BETWEEN 1 AND 5),
    enjoyment        DECIMAL(4, 2)  NOT NULL CHECK (enjoyment BETWEEN 1 AND 5),
    overall_rating   DECIMAL(4, 2)  NOT NULL CHECK (overall_rating   BETWEEN 1 AND 5),
    timestamp        DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (user_id, movie_id),
    FOREIGN KEY (user_id)  REFERENCES Users(user_id)  ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES Movies(movie_id) ON DELETE CASCADE,
    INDEX idx_ratings_user  (user_id),
    INDEX idx_ratings_movie (movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 4: WeightProfiles
-- Stores the AGA-optimised criterion weight vector for each user.
-- One row per user — updated whenever AGA re-runs for that user.
-- This is the core caching table that makes live recommendations fast.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS WeightProfiles (
    user_id       INT           NOT NULL,
    w1_storyline  DECIMAL(6, 4) NOT NULL DEFAULT 0.2000,
    w2_acting     DECIMAL(6, 4) NOT NULL DEFAULT 0.2000,
    w3_visuals    DECIMAL(6, 4) NOT NULL DEFAULT 0.2000,
    w4_emotional  DECIMAL(6, 4) NOT NULL DEFAULT 0.2000,
    w5_enjoyment  DECIMAL(6, 4) NOT NULL DEFAULT 0.2000,
    best_mae      DECIMAL(8, 6) NULL,
    generations   SMALLINT      NULL,
    converged     BOOLEAN       NULL,
    computed_at   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_stale      BOOLEAN       NOT NULL DEFAULT FALSE,

    PRIMARY KEY (user_id),
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 5: RecommendationLog
-- Records every recommendation generation event for monitoring and audit.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS RecommendationLog (
    log_id            INT          NOT NULL AUTO_INCREMENT,
    user_id           INT          NOT NULL,
    recommended_ids   JSON         NOT NULL,
    predicted_scores  JSON         NOT NULL,
    generated_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    weights_used      JSON         NULL,

    PRIMARY KEY (log_id),
    FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE,
    INDEX idx_log_user (user_id),
    INDEX idx_log_date (generated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ─────────────────────────────────────────────────────────────────────────────
-- VERIFICATION QUERY
-- ─────────────────────────────────────────────────────────────────────────────
SELECT
    TABLE_NAME,
    TABLE_ROWS,
    ENGINE,
    TABLE_COMMENT
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'mcrs_db'
ORDER BY TABLE_NAME;
