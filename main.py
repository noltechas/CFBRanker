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
        data = response.json()
        if data:
            print(f"Sample game data structure for week {week}:")
            print(f"Keys available: {list(data[0].keys()) if data else 'No games found'}")
        return data
    else:
        raise Exception(f"Failed to fetch data for year {year}, week {week}: {response.status_code}")

def process_games(data, teams, conferences):
    games_processed = 0
    total_games = len(data)
    completed_games = len([g for g in data if g.get('completed', False)])

    print(f"Processing {total_games} total games, {completed_games} completed")

    # Define known FBS conferences
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

            # Determine if team is FBS based on classification AND conference
            home_is_fbs_eligible = (
                    home_classification == 'fbs' or
                    home_conference in FBS_CONFERENCES
            )
            away_is_fbs_eligible = (
                    away_classification == 'fbs' or
                    away_conference in FBS_CONFERENCES
            )

            # Skip games that don't involve any FBS teams
            if not (home_is_fbs_eligible or away_is_fbs_eligible):
                continue

            # Set proper classification - only mark as FBS if truly FBS
            home_classification = 'fbs' if (home_classification == 'fbs' or home_conference in FBS_CONFERENCES) else (home_classification or 'fcs')
            away_classification = 'fbs' if (away_classification == 'fbs' or away_conference in FBS_CONFERENCES) else (away_classification or 'fcs')

            home_team = teams.get(home_id)
            if not home_team:
                home_conf = conferences.get(home_conference)
                if not home_conf:
                    home_conf = Conference(home_conference)
                    conferences[home_conference] = home_conf
                home_team = Team(home_id, home_team_name, home_conf, home_classification)
                teams[home_id] = home_team
                home_conf.add_team(home_team)

            away_team = teams.get(away_id)
            if not away_team:
                away_conf = conferences.get(away_conference)
                if not away_conf:
                    away_conf = Conference(away_conference)
                    conferences[away_conference] = away_conf
                away_team = Team(away_id, away_team_name, away_conf, away_classification)
                teams[away_id] = away_team
                away_conf.add_team(away_team)

            home_is_fbs = home_team.division.lower() == 'fbs'
            away_is_fbs = away_team.division.lower() == 'fbs'

            if home_is_fbs or away_is_fbs:
                total_points = home_points + away_points
                if total_points > 0:
                    if home_points > away_points:
                        margin = (home_points - away_points) / total_points
                    else:
                        margin = (away_points - home_points) / total_points
                else:
                    margin = 0

                if home_points > away_points:
                    home_team.add_win(away_team.id, away_is_fbs, margin)
                    away_team.add_loss(home_team.id, margin)
                elif away_points > home_points:
                    away_team.add_win(home_team.id, home_is_fbs, margin)
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
    fbs_teams = [team for team in teams.values() if team.division.lower() == 'fbs']

    print(f"Calculating rankings for {len(fbs_teams)} FBS teams with recursion factor {recursion_factor}")

    for i, team in enumerate(fbs_teams, 1):
        cache = {}
        wins, losses = calculate_score(team, recursion_factor, [team.id], cache, mov)
        total_games = team.wins_count + team.losses_count
        if total_games > 0:
            score = (wins - losses) / total_games
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

    # Fixed matplotlib deprecation warning
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
    plt.title(f'All-Time Top {top_n_teams} Team Rankings by Recursion Factor')
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


def detect_conference_championships(year, teams):
    """
    Detect actual conference championship games and their winners
    Returns a dictionary of {conference_name: winning_team}
    """
    championship_winners = {}

    # Check weeks 14-16 for championship games (different conferences play at different times)
    for week in range(14, 17):
        try:
            data = fetch_data(year, week)
            print(f"Checking week {week} for conference championship games...")

            for game in data:
                if not game.get('completed', False):
                    continue

                # Must be a conference game
                if not game.get('conferenceGame', False):
                    continue

                home_team_name = game.get('homeTeam')
                away_team_name = game.get('awayTeam')
                home_conference = game.get('homeConference')
                away_conference = game.get('awayConference')

                # Both teams must be from same conference for it to be a championship
                if not home_conference or home_conference != away_conference:
                    continue

                home_points = game.get('homePoints', 0)
                away_points = game.get('awayPoints', 0)

                if home_points == away_points:  # No ties in championship games
                    continue

                # Find the teams in our data
                home_team = None
                away_team = None
                for team in teams.values():
                    if team.name == home_team_name and team.conference.name == home_conference:
                        home_team = team
                    elif team.name == away_team_name and team.conference.name == away_conference:
                        away_team = team

                if not home_team or not away_team:
                    continue

                # Determine winner
                if home_points > away_points:
                    winner = home_team
                    loser = away_team
                else:
                    winner = away_team
                    loser = home_team

                # Only count as championship if both teams are highly ranked in their conference
                # (to avoid counting random late-season conference games)
                conf_teams = [t for t in teams.values() if t.conference.name == home_conference and t.division.lower() == 'fbs']
                conf_teams.sort(key=lambda t: (t.wins_count, -t.losses_count), reverse=True)

                if len(conf_teams) >= 2 and winner in conf_teams[:4] and loser in conf_teams[:4]:
                    championship_winners[home_conference] = winner
                    print(f"  Found championship: {winner.name} defeated {loser.name} in {home_conference} championship")

        except Exception as e:
            print(f"Error checking week {week} for championships: {e}")
            continue

    return championship_winners

def create_playoff_bracket(final_rankings, conferences, auto_bids, year):
    if not final_rankings:
        print("Cannot create playoff bracket: no teams ranked")
        return

    # Get all teams for championship detection
    all_teams = {team.id: team for team, _ in final_rankings}

    # Step 1: Detect actual conference championship winners
    actual_champions = detect_conference_championships(year, all_teams)

    print(f"\nDetected {len(actual_champions)} actual conference champions:")
    for conf, champ in actual_champions.items():
        print(f"  {conf}: {champ.name}")

    # Step 2: Identify all conference champions (actual winners + highest ranked from other conferences)
    all_conference_champions = []
    conferences_with_champions = set()

    # Start with actual championship winners
    for conf, champ in actual_champions.items():
        champ_score = next((score for team, score in final_rankings if team.id == champ.id), 0)
        all_conference_champions.append((champ, champ_score, "Championship Winner"))
        conferences_with_champions.add(conf)

    # Add highest-ranked teams from conferences without championships
    for team, score in final_rankings:
        conf_name = team.conference.name
        if (conf_name not in ["FBS Independents", "Independent"] and
                conf_name not in conferences_with_champions):
            all_conference_champions.append((team, score, "Conference Leader"))
            conferences_with_champions.add(conf_name)

    # Step 3: Select top 5 conference champions for auto-bids
    all_conference_champions.sort(key=lambda x: x[1], reverse=True)
    auto_bid_champions = all_conference_champions[:5]

    print(f"\nTop 5 Conference Champions (Auto-bids):")
    for i, (team, score, champion_type) in enumerate(auto_bid_champions, 1):
        print(f"  {i}. {team.name} ({team.conference.name}) - {score:.6f} [{champion_type}]")

    # Step 4: Fill remaining 7 spots with highest-ranked teams not already selected
    auto_bid_team_ids = {team.id for team, _, _ in auto_bid_champions}
    at_large_teams = []

    for team, score in final_rankings:
        if team.id not in auto_bid_team_ids and len(at_large_teams) < 7:
            at_large_teams.append((team, score))

    print(f"\nAt-Large Teams (7 spots):")
    for i, (team, score) in enumerate(at_large_teams, 1):
        print(f"  {i}. {team.name} ({team.conference.name}) - {score:.6f}")

    # Step 5: Create final 12-team field and seed by overall ranking
    playoff_teams = []

    # Add auto-bid teams
    for team, score, _ in auto_bid_champions:
        playoff_teams.append((team, score, "Auto-bid"))

    # Add at-large teams
    for team, score in at_large_teams:
        playoff_teams.append((team, score, "At-large"))

    # Sort all 12 teams by ranking for seeding (NOT by auto-bid status)
    playoff_teams.sort(key=lambda x: x[1], reverse=True)

    # Assign seeds 1-12 based purely on ranking
    seeds = {}
    for i, (team, _, _) in enumerate(playoff_teams, 1):
        seeds[team] = i

    print(f"\nFinal 12-team playoff field (seeded by ranking):")
    for i, (team, score, selection_type) in enumerate(playoff_teams, 1):
        bye_status = " [BYE]" if i <= 4 else ""
        print(f"  {i}. {team.name} ({team.conference.name}) - {score:.6f} [{selection_type}]{bye_status}")

    # Step 6: Draw the bracket (using original formatting)
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

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

    # Get teams for bracket positions
    first_round_teams = [team for team, _, _ in playoff_teams[4:]]  # Seeds 5-12
    top_4_teams = [team for team, _, _ in playoff_teams[:4]]  # Seeds 1-4

    if len(first_round_teams) >= 8 and len(top_4_teams) >= 4:
        # First round games
        g1 = draw_game(round_1_pos[0], first_round_teams[0], first_round_teams[7], 0,
                       seeds[first_round_teams[0]], seeds[first_round_teams[7]])
        g2 = draw_game(round_1_pos[1], first_round_teams[1], first_round_teams[6], 0,
                       seeds[first_round_teams[1]], seeds[first_round_teams[6]])
        g3 = draw_game(round_1_pos[2], first_round_teams[2], first_round_teams[5], 0,
                       seeds[first_round_teams[2]], seeds[first_round_teams[5]])
        g4 = draw_game(round_1_pos[3], first_round_teams[3], first_round_teams[4], 0,
                       seeds[first_round_teams[3]], seeds[first_round_teams[4]])

        # Quarterfinals
        q1 = draw_game(quarter_pos[0], top_4_teams[0], None, 1, 1)
        q4 = draw_game(quarter_pos[1], top_4_teams[3], None, 1, 4)
        q2 = draw_game(quarter_pos[2], top_4_teams[1], None, 1, 2)
        q3 = draw_game(quarter_pos[3], top_4_teams[2], None, 1, 3)

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

    ax.text(90, final_pos, "National\nChampionship", ha='center', va='center', fontsize=12)
    ax.text(20, 95, "First Round", ha='center', va='center', fontsize=12)
    ax.text(45, 95, "Quarterfinals", ha='center', va='center', fontsize=12)
    ax.text(70, 95, "Semifinals", ha='center', va='center', fontsize=12)
    ax.text(95, 95, "Final", ha='center', va='center', fontsize=12)
    plt.title(f"College Football Playoff Bracket (5+7 Model)", fontsize=16)
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
        # Pass the year parameter to the bracket function
        create_playoff_bracket(final_rankings, conferences, auto_bids, year)
    else:
        print("No rankings generated - no games found or processed")

if __name__ == "__main__":
    year = 2024
    current_week = 15
    auto_bids = 5
    mov = False
    main(year, current_week, auto_bids, mov)