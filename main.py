import requests
import matplotlib.pyplot as plt
import json
import math

from Conference import Conference
from Team import Team
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
from matplotlib.patches import Rectangle, ConnectionPatch
from professional_bracket import create_espn_style_bracket

API_KEY = "Exhp8ntCBiMjlAyMJea7o7DGhn+u9MVO9z1ziK1RrkyYVIECNXAczguAywpBVNDM"

top_n_teams = 35


def load_team_info():
    with open('team_info.json', 'r') as f:
        return {team['id']: team for team in json.load(f)}


team_info = load_team_info()


def calculate_margin(home_points, away_points, mov_type):
    if home_points == away_points:
        return 0

    winning_points = max(home_points, away_points)
    losing_points = min(home_points, away_points)
    total_points = home_points + away_points

    if mov_type == "logarithmic":
        point_diff = winning_points - losing_points
        if total_points == 0:
            return 0
        margin = math.log(1 + point_diff) / math.log(1 + total_points)
        return margin

    elif mov_type == "capped_linear":
        point_diff = winning_points - losing_points
        capped_diff = min(point_diff, 35)
        margin = 1 + (capped_diff / 35) * 0.5
        return margin

    else:
        if total_points == 0:
            return 0
        return (winning_points - losing_points) / total_points


def fetch_data(year, week):
    url = f"https://api.collegefootballdata.com/games?year={year}&week={week}&seasonType=regular"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            print(f"Sample game data structure for week {week}:")
            print(f"Keys available: {list(data[0].keys()) if data else 'No games found'}")
        return data
    else:
        raise Exception(f"Failed to fetch data for year {year}, week {week}: {response.status_code}")

def process_games(data, teams, conferences, mov_type="none"):
    games_processed = 0
    total_games = len(data)
    completed_games = len([g for g in data if g.get('completed', False)])

    print(f"Processing {total_games} total games, {completed_games} completed")
    print(f"Using MOV algorithm: {mov_type}")

    FBS_CONFERENCES = {
        'SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12', 'American Athletic',
        'Mountain West', 'Conference USA', 'Mid-American', 'Sun Belt',
        'FBS Independents', 'Independent'
    }

    for game in data:
        try:
            if not game.get('completed', False):
                continue

            home_team_name = game.get('homeTeam')
            away_team_name = game.get('awayTeam')

            if not home_team_name or not away_team_name:
                continue

            home_id = game.get('homeId', hash(home_team_name))
            away_id = game.get('awayId', hash(away_team_name))

            home_points = game.get('homePoints')
            away_points = game.get('awayPoints')

            if home_points is None or away_points is None:
                continue

            if home_points == 0 and away_points == 0:
                continue

            home_conference = game.get('homeConference') or "Independent"
            away_conference = game.get('awayConference') or "Independent"
            home_classification = game.get('homeClassification')
            away_classification = game.get('awayClassification')

            home_is_fbs = (
                    home_classification == 'fbs' and
                    (home_conference in FBS_CONFERENCES or home_conference == "Independent")
            )
            away_is_fbs = (
                    away_classification == 'fbs' and
                    (away_conference in FBS_CONFERENCES or away_conference == "Independent")
            )

            if not (home_is_fbs or away_is_fbs):
                continue

            if not home_is_fbs:
                home_classification = home_classification or 'fcs'
            if not away_is_fbs:
                away_classification = away_classification or 'fcs'

            home_team = teams.get(home_id)
            if not home_team:
                home_conf = conferences.get(home_conference)
                if not home_conf:
                    home_conf = Conference(home_conference)
                    conferences[home_conference] = home_conf
                final_home_classification = 'fbs' if home_is_fbs else (home_classification or 'fcs')
                home_team = Team(home_id, home_team_name, home_conf, final_home_classification)
                teams[home_id] = home_team
                home_conf.add_team(home_team)

            away_team = teams.get(away_id)
            if not away_team:
                away_conf = conferences.get(away_conference)
                if not away_conf:
                    away_conf = Conference(away_conference)
                    conferences[away_conference] = away_conf
                final_away_classification = 'fbs' if away_is_fbs else (away_classification or 'fcs')
                away_team = Team(away_id, away_team_name, away_conf, final_away_classification)
                teams[away_id] = away_team
                away_conf.add_team(away_team)

            home_team_is_fbs = home_team.division.lower() == 'fbs'
            away_team_is_fbs = away_team.division.lower() == 'fbs'

            if home_team_is_fbs or away_team_is_fbs:
                margin = calculate_margin(home_points, away_points, mov_type)

                if home_points > away_points:
                    home_team.add_win(away_team.id, away_team_is_fbs, margin)
                    away_team.add_loss(home_team.id, margin)
                elif away_points > home_points:
                    away_team.add_win(home_team.id, home_team_is_fbs, margin)
                    home_team.add_loss(away_team.id, margin)

                games_processed += 1

        except Exception as e:
            print(f"Error processing game: {e}")
            continue

    print(f"Successfully processed {games_processed} completed games")
    print(f"Created {len(teams)} teams, {len([t for t in teams.values() if t.division.lower() == 'fbs'])} FBS")
    return teams, conferences

def calculate_ranking(teams, recursion_factor, mov=True):
    def calculate_score(team, depth, visited_team_ids, cache, mov):
        if depth == 0:
            return team.fbs_wins_count, team.losses_count

        cache_key = (team.id, depth)
        if cache_key in cache:
            return cache[cache_key]

        wins, losses = 0, 0
        for opponent_id, count in team.wins.items():
            if opponent_id not in visited_team_ids and opponent_id in teams:
                w, l = calculate_score(teams[opponent_id], depth - 1, visited_team_ids + [opponent_id], cache, mov)
                if mov:
                    wins += w * sum(item[0] * item[1] for item in count)
                else:
                    wins += w * sum(item[0] for item in count)
        for opponent_id, count in team.losses.items():
            if opponent_id not in visited_team_ids and opponent_id in teams:
                w, l = calculate_score(teams[opponent_id], depth - 1, visited_team_ids + [opponent_id], cache, mov)
                if mov:
                    losses += l * sum(item[0] * item[1] for item in count)
                else:
                    losses += l * sum(item[0] for item in count)
        cache[cache_key] = (wins, losses)
        return wins, losses

    rankings = {}
    fbs_teams = [team for team in teams.values() if team.division.lower() == 'fbs' and
                 (team.wins_count > 0 or team.losses_count > 0)]

    print(f"Calculating rankings for {len(fbs_teams)} FBS teams with recursion factor {recursion_factor}")

    for i, team in enumerate(fbs_teams, 1):
        cache = {}
        wins, losses = calculate_score(team, recursion_factor, [team.id], cache, mov)
        total_games = team.wins_count + team.losses_count
        if total_games > 0:
            score = (wins - losses) / total_games
            if score > 0:
                rankings[team] = score
            if i <= 10:
                print(f"  {team.name}: wins={wins:.2f}, losses={losses:.2f}, games={total_games}, score={score:.6f}")
        else:
            rankings[team] = 0

    return rankings

def rank_teams(rankings):
    return sorted(rankings.items(), key=lambda x: x[1], reverse=True)

def get_team_color(team_id):
    team_data = team_info.get(team_id)
    if team_data:
        return team_data.get('color', '#000000')
    return '#000000'


def get_team_logo(team_id):
    team_data = team_info.get(team_id)
    if team_data and team_data.get('logos'):
        logo_url = team_data['logos'][0]
        try:
            response = requests.get(logo_url)
            img = Image.open(BytesIO(response.content))
            return img
        except:
            return None
    return None


def plot_rankings(all_rankings):
    if not all_rankings or not any(all_rankings):
        print("No rankings data to plot")
        return

    final_rankings = rank_teams(all_rankings[-1])

    if not final_rankings:
        print("No teams in final rankings")
        return

    plt.figure(figsize=(20, 12))

    min_score = min(score for _, score in final_rankings)
    max_score = max(score for _, score in final_rankings)

    def normalize_score(score):
        if max_score == min_score:
            return 1.0
        return round((score - min_score) / (max_score - min_score), 3)

    normalized_scores = {team: normalize_score(score) for team, score in final_rankings}

    ever_top_n_teams = set()
    for rankings in all_rankings:
        if rankings:
            ever_top_n_teams.update(team for team, _ in rank_teams(rankings)[:top_n_teams])

    final_top_n = [team for team, _ in final_rankings if team in ever_top_n_teams][:top_n_teams]

    color_map = plt.colormaps.get_cmap('tab20')
    team_colors = {team: color_map(i / max(len(ever_top_n_teams), 1)) for i, team in enumerate(ever_top_n_teams)}

    for team in ever_top_n_teams:
        rankings = []
        for r in all_rankings:
            if r:
                rank = next((i for i, (t, _) in enumerate(rank_teams(r), 1) if t == team), None)
                rankings.append(rank if rank and rank <= top_n_teams else None)
            else:
                rankings.append(None)

        color = get_team_color(team.id) or team_colors[team]
        line, = plt.plot(range(len(all_rankings)), rankings, marker='o', color=color, alpha=0.7)

        logo = get_team_logo(team.id)
        if logo:
            logo = logo.resize((50, 50))
            for x, y in enumerate(rankings):
                if y is not None:
                    im = OffsetImage(logo, zoom=0.5)
                    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                    plt.gca().add_artist(ab)

    plt.xlabel('Recursion Factor')
    plt.ylabel('Ranking')
    plt.title(f'Top {top_n_teams} Team Rankings by Recursion Factor')
    plt.ylim(top_n_teams + 0.5, 0.5)
    plt.xlim(-0.5, len(all_rankings) - 0.5)

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    for i, team in enumerate(final_top_n):
        y = 1 - (i + 0.5) / max(len(final_top_n), 1)
        logo = get_team_logo(team.id)
        if logo:
            logo = logo.resize((30, 30))
            im = OffsetImage(logo, zoom=1)
            ab = AnnotationBbox(im, (1.02, y), xycoords='axes fraction', box_alignment=(0, 0.5), frameon=False)
            ax.add_artist(ab)

        team_label = f"{i + 1}. {team.name} ({team.wins_count}-{team.losses_count}) {normalized_scores[team]:.3f}"
        ax.text(1.06, y, team_label, transform=ax.transAxes, va='center')

    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f'all_time_top_{top_n_teams}_team_rankings_with_normalized_scores.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'all_time_top_{top_n_teams}_team_rankings_with_normalized_scores.png'")


def main(year, current_week, auto_bids=5, mov=True, mov_type="none"):
    teams = {}
    conferences = {}

    for week in range(1, current_week + 1):
        try:
            data = fetch_data(year, week)
            teams, conferences = process_games(data, teams, conferences, mov_type if mov else "none")
            print(f"Processed data for year {year}, week {week}")
        except Exception as e:
            print(f"Error processing week {week}: {e}")

    print(f"\nFound {len(teams)} total teams, {len([t for t in teams.values() if t.division.lower() == 'fbs'])} FBS teams")

    print("\nTeam records:")
    fbs_teams_with_games = [team for team in teams.values() if team.division.lower() == 'fbs' and (team.wins_count > 0 or team.losses_count > 0)]

    for team in sorted(fbs_teams_with_games, key=lambda t: (t.wins_count, -t.losses_count), reverse=True)[:10]:
        print(f"{team} ({team.conference})")
        if team.wins:
            print(f"  Wins against: {', '.join(f'{teams[opponent_id].name}' for opponent_id in team.wins.keys() if opponent_id in teams)}")
        if team.losses:
            print(f"  Losses against: {', '.join(f'{teams[opponent_id].name}' for opponent_id in team.losses.keys() if opponent_id in teams)}")

    all_rankings = []
    for recursion_factor in range(current_week):
        rankings = calculate_ranking(teams, recursion_factor, mov)
        all_rankings.append(rankings)

        ranked_teams = rank_teams(rankings)

        print(f"\nTop {min(top_n_teams, len(ranked_teams))} Rankings (Recursion Factor: {recursion_factor}):")
        for i, (team, score) in enumerate(ranked_teams[:top_n_teams], 1):
            print(f"{i}. {team} ({team.conference}): {score:.6f}")

    if any(all_rankings):
        plot_rankings(all_rankings)
        final_rankings = rank_teams(all_rankings[-1])
        create_espn_style_bracket(final_rankings, conferences, auto_bids, year, team_info)
    else:
        print("No rankings generated - no games found or processed")

if __name__ == "__main__":
    year = 2024
    current_week = 15
    auto_bids = 5
    mov = True
    mov_type = "logarithmic"  # Options: "none", "logarithmic", "capped_linear"
    main(year, current_week, auto_bids, mov, mov_type)