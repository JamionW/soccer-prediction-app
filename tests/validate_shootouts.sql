-- SQL Script to Validate Shootout Data Quality Issues
-- Run these queries to find problematic games in your database

-- ===== QUERY 1: Games marked as shootout but have different regulation scores =====
-- These should NOT be shootouts if the regulation score is not tied
SELECT 
    g.game_id,
    g.date,
    ht.team_name as home_team,
    at.team_name as away_team,
    g.home_score,
    g.away_score,
    g.home_penalties,
    g.away_penalties,
    g.went_to_shootout,
    g.asa_loaded,
    g.season_year,
    'INVALID_SHOOTOUT_DIFFERENT_SCORES' as issue_type
FROM games g
JOIN team ht ON g.home_team_id = ht.team_id
JOIN team at ON g.away_team_id = at.team_id
WHERE g.went_to_shootout = true 
  AND g.home_score != g.away_score
  AND g.is_completed = true
  AND g.season_year = 2025
ORDER BY g.date DESC;

-- ===== QUERY 2: Games marked as shootout but missing penalty data =====
-- These have shootout flag but no penalty scores
SELECT 
    g.game_id,
    g.date,
    ht.team_name as home_team,
    at.team_name as away_team,
    g.home_score,
    g.away_score,
    g.home_penalties,
    g.away_penalties,
    g.went_to_shootout,
    g.asa_loaded,
    'SHOOTOUT_MISSING_PENALTY_DATA' as issue_type
FROM games g
JOIN team ht ON g.home_team_id = ht.team_id
JOIN team at ON g.away_team_id = at.team_id
WHERE g.went_to_shootout = true 
  AND (g.home_penalties IS NULL OR g.away_penalties IS NULL 
       OR (g.home_penalties = 0 AND g.away_penalties = 0))
  AND g.is_completed = true
  AND g.season_year = 2025
ORDER BY g.date DESC;

-- ===== QUERY 3: Games with penalty data but not marked as shootout =====
-- These might be missing the shootout flag
SELECT 
    g.game_id,
    g.date,
    ht.team_name as home_team,
    at.team_name as away_team,
    g.home_score,
    g.away_score,
    g.home_penalties,
    g.away_penalties,
    g.went_to_shootout,
    g.asa_loaded,
    'HAS_PENALTIES_NOT_MARKED_SHOOTOUT' as issue_type
FROM games g
JOIN team ht ON g.home_team_id = ht.team_id
JOIN team at ON g.away_team_id = at.team_id
WHERE g.went_to_shootout = false 
  AND (g.home_penalties > 0 OR g.away_penalties > 0)
  AND g.is_completed = true
  AND g.season_year = 2025
ORDER BY g.date DESC;

-- ===== QUERY 4: All shootout games for specific problematic teams =====
-- Focus on RBNY II and NYCFC II to see their shootout games
SELECT 
    g.game_id,
    g.date,
    CASE 
        WHEN g.home_team_id IN ('9Yqdwg85vJ', 'jYQJXkP5GR') THEN 'HOME'
        ELSE 'AWAY' 
    END as team_location,
    ht.team_name as home_team,
    at.team_name as away_team,
    g.home_score,
    g.away_score,
    g.home_penalties,
    g.away_penalties,
    g.went_to_shootout,
    g.asa_loaded,
    g.status,
    -- Calculate who won the shootout
    CASE 
        WHEN g.went_to_shootout AND g.home_penalties > g.away_penalties THEN ht.team_name
        WHEN g.went_to_shootout AND g.away_penalties > g.home_penalties THEN at.team_name
        WHEN g.went_to_shootout THEN 'TIE_OR_INVALID'
        ELSE 'NO_SHOOTOUT'
    END as shootout_winner
FROM games g
JOIN team ht ON g.home_team_id = ht.team_id
JOIN team at ON g.away_team_id = at.team_id
WHERE (g.home_team_id IN ('9Yqdwg85vJ', 'jYQJXkP5GR') 
       OR g.away_team_id IN ('9Yqdwg85vJ', 'jYQJXkP5GR'))
  AND g.went_to_shootout = true
  AND g.is_completed = true
  AND g.season_year = 2025
ORDER BY g.date DESC;

-- ===== QUERY 5: Summary of data quality issues =====
-- Get counts of each type of issue
SELECT 
    'Games marked shootout with different scores' as issue_type,
    COUNT(*) as count
FROM games g
WHERE g.went_to_shootout = true 
  AND g.home_score != g.away_score
  AND g.is_completed = true
  AND g.season_year = 2025

UNION ALL

SELECT 
    'Games marked shootout missing penalty data' as issue_type,
    COUNT(*) as count
FROM games g
WHERE g.went_to_shootout = true 
  AND (g.home_penalties IS NULL OR g.away_penalties IS NULL 
       OR (g.home_penalties = 0 AND g.away_penalties = 0))
  AND g.is_completed = true
  AND g.season_year = 2025

UNION ALL

SELECT 
    'Games with penalties not marked as shootout' as issue_type,
    COUNT(*) as count
FROM games g
WHERE g.went_to_shootout = false 
  AND (g.home_penalties > 0 OR g.away_penalties > 0)
  AND g.is_completed = true
  AND g.season_year = 2025

UNION ALL

SELECT 
    'Total shootout games in 2025' as issue_type,
    COUNT(*) as count
FROM games g
WHERE g.went_to_shootout = true
  AND g.is_completed = true
  AND g.season_year = 2025;

-- ===== QUERY 6: Specific problematic game from RBNY II =====
-- The 2-5 game that was marked as shootout
SELECT 
    g.*,
    ht.team_name as home_team,
    at.team_name as away_team
FROM games g
JOIN team ht ON g.home_team_id = ht.team_id
JOIN team at ON g.away_team_id = at.team_id
WHERE (g.home_team_id = '9Yqdwg85vJ' OR g.away_team_id = '9Yqdwg85vJ')
  AND g.home_score = 2 
  AND g.away_score = 5
  AND g.season_year = 2025
  AND g.is_completed = true;

-- ===== QUERY 7: All RBNY II games for manual verification =====
-- Get all completed games for RBNY II to manually check against ASA
SELECT 
    g.game_id,
    g.date,
    CASE 
        WHEN g.home_team_id = '9Yqdwg85vJ' THEN 'vs ' || at.team_name
        ELSE '@ ' || ht.team_name
    END as matchup,
    CASE 
        WHEN g.home_team_id = '9Yqdwg85vJ' THEN g.home_score || '-' || g.away_score
        ELSE g.away_score || '-' || g.home_score
    END as score_from_rbny_perspective,
    g.went_to_shootout,
    CASE 
        WHEN g.went_to_shootout THEN g.home_penalties || '-' || g.away_penalties
        ELSE 'No shootout'
    END as penalty_score,
    g.asa_loaded,
    g.status
FROM games g
JOIN team ht ON g.home_team_id = ht.team_id
JOIN team at ON g.away_team_id = at.team_id
WHERE (g.home_team_id = '9Yqdwg85vJ' OR g.away_team_id = '9Yqdwg85vJ')
  AND g.is_completed = true
  AND g.season_year = 2025
ORDER BY g.date;