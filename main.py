import requests
import matplotlib.pyplot as plt
import json

from Conference import Conference
from Team import Team
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from io import BytesIO
from matplotlib.patches import Rectangle, ConnectionPatch

API_KEY = "Exhp8ntCBiMjlAyMJea7o7DGhn+u9MVO9z1ziK1RrkyYVIECNXAczguAywpBVNDM"

top_n_teams = 35


def load_team_info():
    with open('team_info.json', 'r') as f:
        return {team['id']: team for team in json.load(f)}


team_info = load_team_info()


def fetch_data(year, week):
    url = f"https://api.collegefootballdata.com/games?year={year}&week={week}&seasonType=regular"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data for year {year}, week {week}: {response.status_code}")

def process_games(data, teams, conferences):
    for game in data:
        if game['completed']:
            home_team = teams.get(game['home_id'])
            if not home_team:
                home_conference = conferences.get(game['home_conference'])
                if not home_conference:
                    home_conference = Conference(game['home_conference'])
                    conferences[game['home_conference']] = home_conference
                home_team = Team(game['home_id'], game['home_team'], home_conference, game['home_division'])
                teams[game['home_id']] = home_team
                home_conference.add_team(home_team)

            away_team = teams.get(game['away_id'])
            if not away_team:
                away_conference = conferences.get(game['away_conference'])
                if not away_conference:
                    away_conference = Conference(game['away_conference'])
                    conferences[game['away_conference']] = away_conference
                away_team = Team(game['away_id'], game['away_team'], away_conference, game['away_division'])
                teams[game['away_id']] = away_team
                away_conference.add_team(away_team)

            home_is_fbs = home_team.division == 'fbs'
            away_is_fbs = away_team.division == 'fbs'

            if home_is_fbs or away_is_fbs:
                margin = max(game['home_points'], game['away_points']) / (game['away_points'] + game['home_points'])
                if game['home_points'] > game['away_points']:
                    home_team.add_win(away_team.id, away_is_fbs, margin)
                    away_team.add_loss(home_team.id, margin)
                elif game['away_points'] > game['home_points']:
                    away_team.add_win(home_team.id, home_is_fbs, margin)
                    home_team.add_loss(away_team.id, margin)

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
            if opponent_id not in visited_team_ids:
                w, l = calculate_score(teams[opponent_id], depth - 1, visited_team_ids + [opponent_id], cache, mov)
                if mov:
                    wins += (w * sum(item[0] for item in count)) * (sum(item[1] for item in count))
                else:
                    wins += (w * sum(item[0] for item in count))
        for opponent_id, count in team.losses.items():
            if opponent_id not in visited_team_ids:
                w, l = calculate_score(teams[opponent_id], depth - 1, visited_team_ids + [opponent_id], cache, mov)
                if mov:
                    losses += (l * sum(item[0] for item in count)) * (sum(item[1] for item in count))
                else:
                    losses += (l * sum(item[0] for item in count))
        cache[cache_key] = (wins, losses)
        return wins, losses

    rankings = {}
    fbs_teams = [team for team in teams.values() if team.division == 'fbs']

    for i, team in enumerate(fbs_teams, 1):
        cache = {}
        wins, losses = calculate_score(team, recursion_factor, [team.id], cache, mov)
        total_games = team.wins_count + team.losses_count
        if total_games > 0:
            rankings[team] = (wins - losses) / total_games
        else:
            rankings[team] = 0  # Handle the case where a team hasn't played any games

    return rankings

def rank_teams(rankings):
    return sorted(rankings.items(), key=lambda x: x[1], reverse=True)

def get_team_color(team_id):
    team_data = team_info.get(team_id)
    if team_data:
        return team_data.get('color', '#000000')  # Default to black if color not found
    return '#000000'


def get_team_logo(team_id):
    team_data = team_info.get(team_id)
    if team_data and team_data.get('logos'):
        logo_url = team_data['logos'][0]  # Use the first logo
        response = requests.get(logo_url)
        img = Image.open(BytesIO(response.content))
        return img
    return None


def plot_rankings(all_rankings):
    plt.figure(figsize=(20, 12))

    # Get final rankings for all teams
    final_rankings = rank_teams(all_rankings[-1])

    # Find the minimum and maximum scores
    min_score = min(score for _, score in final_rankings)
    max_score = max(score for _, score in final_rankings)

    # Normalize scores from 0 to 1
    def normalize_score(score):
        if max_score == min_score:
            return 1.0  # Handle the case where all scores are the same
        return round((score - min_score) / (max_score - min_score), 3)

    normalized_scores = {team: normalize_score(score) for team, score in final_rankings}

    # Find teams that were ever in the top N
    ever_top_n_teams = set()
    for rankings in all_rankings:
        ever_top_n_teams.update(team for team, _ in rank_teams(rankings)[:top_n_teams])

    final_top_n = [team for team, _ in final_rankings if team in ever_top_n_teams][:top_n_teams]

    # Create a color map for all teams
    color_map = plt.cm.get_cmap('tab20')
    team_colors = {team: color_map(i / len(ever_top_n_teams)) for i, team in enumerate(ever_top_n_teams)}

    # Plot rankings for all teams that were ever in top N
    for team in ever_top_n_teams:
        rankings = []
        for r in all_rankings:
            rank = next((i for i, (t, _) in enumerate(rank_teams(r), 1) if t == team), None)
            rankings.append(rank if rank and rank <= top_n_teams else None)

        color = get_team_color(team.id) or team_colors[team]
        line, = plt.plot(range(len(all_rankings)), rankings, marker='o', color=color, alpha=0.7)

        # Add logo to each data point
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
    plt.title(f'All-Time Top {top_n_teams} Team Rankings by Recursion Factor')
    plt.ylim(top_n_teams + 0.5, 0.5)
    plt.xlim(-0.5, len(all_rankings) - 0.5)

    # Add logos and team names with records and normalized scores to the right side of the plot
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    for i, team in enumerate(final_top_n):
        y = 1 - (i + 0.5) / len(final_top_n)
        logo = get_team_logo(team.id)
        if logo:
            logo = logo.resize((30, 30))
            im = OffsetImage(logo, zoom=1)
            ab = AnnotationBbox(im, (1.02, y), xycoords='axes fraction', box_alignment=(0, 0.5), frameon=False)
            ax.add_artist(ab)

        # Add team name with record and normalized score
        team_label = f"{i + 1}. {team.name} ({team.wins_count}-{team.losses_count}) {normalized_scores[team]:.3f}"
        ax.text(1.06, y, team_label, transform=ax.transAxes, va='center')

    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f'all_time_top_{top_n_teams}_team_rankings_with_normalized_scores.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'all_time_top_{top_n_teams}_team_rankings_with_normalized_scores.png'")


def create_playoff_bracket(final_rankings, conferences, auto_bids):
    # Ensure auto_bids is at least 4 and at most 12
    auto_bids = max(4, min(auto_bids, 12))

    # Identify the top auto_bids conference champions
    conference_champions = []
    at_large = []
    conferences_seen = set()

    for team, score in final_rankings:
        if team.conference.name not in conferences_seen and len(conference_champions) < auto_bids:
            if team.conference.name != "FBS Independents":
                conference_champions.append((team, score))
            conferences_seen.add(team.conference.name)
        elif len(at_large) < (12 - auto_bids):
            at_large.append((team, score))

        if len(conference_champions) == auto_bids and len(at_large) == (12 - auto_bids):
            break

    # Ensure we have enough teams to fill the bracket
    if len(conference_champions) < auto_bids or len(at_large) < (12 - auto_bids):
        raise ValueError(f"Not enough teams to fill the bracket with {auto_bids} auto-bids")

    # Determine the top 4 seeds (always conference champions) and the remaining seeds
    top_4_seeds = conference_champions[:4]
    remaining_seeds = conference_champions[4:] + at_large

    # Sort remaining seeds by ranking
    team_scores = dict(final_rankings)
    remaining_seeds.sort(key=lambda x: team_scores[x[0]], reverse=True)

    # Assign seed numbers
    seeds = dict(zip([team for team, _ in top_4_seeds], range(1, 5)))
    seeds.update(dict(zip([team for team, _ in remaining_seeds], range(5, 13))))

    # Now proceed with drawing the bracket
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Define positions for each round
    round_1_pos = [85, 65, 45, 25]
    quarter_pos = round_1_pos
    semi_pos = [75, 35]
    final_pos = 55

    def draw_game(y_pos, left_team, right_team, round_num, seed_left=None, seed_right=None):
        rect = Rectangle((10 + round_num * 25, y_pos - 5), 20, 10, fill=False)
        ax.add_patch(rect)
        for team, y_offset, seed in [(left_team, 2, seed_left), (right_team, -2, seed_right)]:
            if team:
                logo = get_team_logo(team.id)
                if logo:
                    logo = logo.resize((60, 60))
                    im = OffsetImage(logo, zoom=0.5)
                    ab = AnnotationBbox(im, (11 + round_num * 25, y_pos + y_offset), xycoords='data', frameon=False)
                    ax.add_artist(ab)
                seed_text = f"({seed}) " if seed is not None else ""
                ax.text(13 + round_num * 25, y_pos + y_offset, f"{seed_text}{team.name}", fontsize=8, va='center')
        return (10 + round_num * 25 + 20, y_pos)

    def draw_connection(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-", color="gray"))

    # First round
    g1 = draw_game(round_1_pos[0], remaining_seeds[0][0], remaining_seeds[7][0], 0, seeds[remaining_seeds[0][0]],
                   seeds[remaining_seeds[7][0]])
    g2 = draw_game(round_1_pos[1], remaining_seeds[3][0], remaining_seeds[4][0], 0, seeds[remaining_seeds[3][0]],
                   seeds[remaining_seeds[4][0]])
    g3 = draw_game(round_1_pos[2], remaining_seeds[2][0], remaining_seeds[5][0], 0, seeds[remaining_seeds[2][0]],
                   seeds[remaining_seeds[5][0]])
    g4 = draw_game(round_1_pos[3], remaining_seeds[1][0], remaining_seeds[6][0], 0, seeds[remaining_seeds[1][0]],
                   seeds[remaining_seeds[6][0]])

    # Quarterfinals
    q1 = draw_game(quarter_pos[0], top_4_seeds[0][0], None, 1, 1)
    q4 = draw_game(quarter_pos[1], top_4_seeds[3][0], None, 1, 4)
    q2 = draw_game(quarter_pos[2], top_4_seeds[1][0], None, 1, 2)
    q3 = draw_game(quarter_pos[3], top_4_seeds[2][0], None, 1, 3)

    # Semifinals and Final
    s1 = draw_game(semi_pos[0], None, None, 2)
    s2 = draw_game(semi_pos[1], None, None, 2)
    f = draw_game(final_pos, None, None, 3)

    # Draw connections
    draw_connection(g1, (q1[0] - 20, q1[1]))
    draw_connection(g2, (q4[0] - 20, q4[1]))
    draw_connection(g3, (q2[0] - 20, q2[1]))
    draw_connection(g4, (q3[0] - 20, q3[1]))
    draw_connection(q1, (s1[0] - 20, s1[1]))
    draw_connection(q4, (s1[0] - 20, s1[1]))
    draw_connection(q2, (s2[0] - 20, s2[1]))
    draw_connection(q3, (s2[0] - 20, s2[1]))
    draw_connection(s1, (f[0] - 20, f[1]))
    draw_connection(s2, (f[0] - 20, f[1]))

    # Add labels and title
    ax.text(90, final_pos, "National\nChampionship", ha='center', va='center', fontsize=12)
    ax.text(20, 95, "First Round", ha='center', va='center', fontsize=12)
    ax.text(45, 95, "Quarterfinals", ha='center', va='center', fontsize=12)
    ax.text(70, 95, "Semifinals", ha='center', va='center', fontsize=12)
    ax.text(95, 95, "Final", ha='center', va='center', fontsize=12)
    plt.title(f"College Football Playoff Bracket ({auto_bids}+{12 - auto_bids} Model)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'playoff_bracket.png', dpi=300, bbox_inches='tight')
    print(f"Enhanced playoff bracket saved as 'playoff_bracket.png'")


def main(year, current_week, auto_bids=5, mov=True):
    teams = {}
    conferences = {}

    for week in range(1, current_week + 1):
        try:
            data = fetch_data(year, week)
            teams, conferences = process_games(data, teams, conferences)
            print(f"Processed data for year {year}, week {week}")
        except Exception as e:
            print(f"Error processing week {week}: {e}")

    print("\nTeam records:")
    for team in sorted(teams.values(), key=lambda t: (t.wins_count, -t.losses_count), reverse=True):
        if team.division == 'fbs':
            print(f"{team} ({team.conference})")
            print(
                f"  Wins against: {', '.join(f'{teams[opponent_id].name} (x{count})' for opponent_id, count in team.wins.items())}")
            print(
                f"  Losses against: {', '.join(f'{teams[opponent_id].name} (x{count})' for opponent_id, count in team.losses.items())}")

    all_rankings = []
    for recursion_factor in range(current_week):
        rankings = calculate_ranking(teams, recursion_factor, mov)
        all_rankings.append(rankings)

        ranked_teams = rank_teams(rankings)

        print(f"\nTop {top_n_teams} Rankings (Recursion Factor: {recursion_factor}):")
        for i, (team, score) in enumerate(ranked_teams[:top_n_teams], 1):
            print(f"{i}. {team} ({team.conference}): {score}")

    plot_rankings(all_rankings)

    # Create playoff bracket based on final rankings
    final_rankings = rank_teams(all_rankings[-1])
    create_playoff_bracket(final_rankings, conferences, auto_bids)

if __name__ == "__main__":
    year = 2024
    current_week = 7
    auto_bids = 5
    mov = False
    main(year, current_week, auto_bids, mov)