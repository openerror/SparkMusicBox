CREATE DATABASE IF NOT EXISTS bittiger_capstone;

USE bittiger_capstone;

DROP TABLE IF EXISTS Song;
DROP TABLE IF EXISTS Rating;
DROP TABLE IF EXISTS Recommendation;

-- Records music metadata
-- Do not create unless there's actual data, or else table Rating would fail to be populated
-- CREATE TABLE  IF NOT EXISTS Song
-- (
--   song_id int,
--   singer varchar(255),
--   song_name varchar(255)
--   PRIMARY KEY(song_id)
-- );

-- Records existing implicit ratings
CREATE TABLE  IF NOT EXISTS Rating
(
  uid int,
  song_id int,
  rating float,
  PRIMARY KEY(uid, song_id)
  -- FOREIGN KEY(song_id)
    -- REFERENCES Song(song_id)
);

-- Stores recommendations made by Spark ALS
CREATE TABLE  IF NOT EXISTS Recommendation
(
  uid int,
  song_id int,
  prediction float,
  PRIMARY KEY(uid, song_id)
  -- FOREIGN KEY(song_id)
    -- REFERENCES Song(song_id)
);
