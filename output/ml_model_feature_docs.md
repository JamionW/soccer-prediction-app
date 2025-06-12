# AutoGluon Model Feature Documentation

## Feature Descriptions

### Team Identity
- **is_home**: Binary flag (1 if team is playing at home, 0 if away)

### Expected Goals (xG) Features
- **team_xgf_per_game**: Team's expected goals for per game
- **team_xga_per_game**: Team's expected goals against per game
- **opp_xgf_per_game**: Opponent's expected goals for per game
- **opp_xga_per_game**: Opponent's expected goals against per game
- **xg_diff**: Team's xG differential (xGF - xGA)
- **opp_xg_diff**: Opponent's xG differential

### Recent Form (Last 5 games)
- **team_form_points**: Team's points per game in last 5 games
- **team_form_gf**: Team's goals for per game in last 5
- **team_form_ga**: Team's goals against per game in last 5
- **opp_form_points**: Opponent's points per game in last 5
- **opp_form_gf**: Opponent's goals for per game in last 5
- **opp_form_ga**: Opponent's goals against per game in last 5

### Head-to-Head History
- **h2h_games_played**: Number of previous meetings
- **h2h_win_rate**: Team's win rate vs this opponent
- **h2h_goals_for_avg**: Team's average goals scored in H2H
- **h2h_goals_against_avg**: Team's average goals conceded in H2H

### Match Context
- **team_rest_days**: Days since team's last game
- **opp_rest_days**: Days since opponent's last game
- **month**: Month of the game (1-12)
- **day_of_week**: Day of week (0=Monday, 6=Sunday)
- **is_weekend**: Binary flag (1 if Fri/Sat/Sun, 0 otherwise)

## Usage Notes

1. The model predicts goals for the 'team' (not home/away)
2. Set is_home=1 when predicting for home team
3. All stats should be calculated BEFORE the game being predicted
4. Form metrics use the 5 most recent games
5. Rest days are capped at reasonable values (e.g., 14 days)
