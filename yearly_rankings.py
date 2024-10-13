# weekly_rankings.py

import matplotlib.pyplot as plt
from main import fetch_data, process_games, calculate_ranking, rank_teams, get_team_logo, get_team_color
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def calculate_weekly_rankings(year, mov):
    teams = {}
    conferences = {}
    weekly_rankings = []

    for week in range(1, 15):  # 14 weeks in a regular season
        try:
            data = fetch_data(year, week)
            teams, conferences = process_games(data, teams, conferences)

            rankings = calculate_ranking(teams, week - 1, mov)  # Use week as recursion factor
            ranked_teams = rank_teams(rankings)

            # Filter out teams with score <= 0
            positive_ranked_teams = [(team, score) for team, score in ranked_teams if score > 0]

            weekly_rankings.append(positive_ranked_teams)
            print(f"Processed rankings for week {week}")
        except Exception as e:
            print(f"Error processing week {week}: {e}")

    return weekly_rankings


def plot_weekly_rankings(weekly_rankings):
    plt.figure(figsize=(20, 12))

    all_ranked_teams = set()
    for week_ranking in weekly_rankings:
        all_ranked_teams.update(team for team, _ in week_ranking[:25])  # Consider top 25 from each week

    final_rankings = weekly_rankings[-1]

    for team in all_ranked_teams:
        rankings = []
        for week, week_ranking in enumerate(weekly_rankings):
            rank = next((i for i, (t, _) in enumerate(week_ranking[:25], 1) if t == team), None)
            rankings.append(rank)

        color = get_team_color(team.id)
        plt.plot(range(1, len(weekly_rankings) + 1), rankings, marker='o', label=team.name, color=color)

        # Add larger logo to each data point
        logo = get_team_logo(team.id)
        if logo:
            logo = logo.resize((40, 40))  # Increased from (20, 20) to (40, 40)
            for x, y in enumerate(rankings, 1):
                if y is not None:
                    im = OffsetImage(logo, zoom=0.5)  # Increased zoom from 0.5 to 1
                    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                    plt.gca().add_artist(ab)

    plt.gca().invert_yaxis()  # Invert y-axis so that rank 1 is at the top
    plt.xlabel('Week')
    plt.ylabel('Ranking')
    plt.title('Top 25 Team Rankings Throughout the Season')
    plt.ylim(25.5, 0.5)
    plt.xlim(0.5, len(weekly_rankings) + 0.5)

    # Adjust the plot layout to make room for the legend
    plt.subplots_adjust(right=0.75)

    # Add legend with final top 25 rankings
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    final_top_25 = final_rankings[:25]
    for i, (team, score) in enumerate(final_top_25):
        y = 1 - (i + 0.5) / len(final_top_25)
        logo = get_team_logo(team.id)
        if logo:
            logo = logo.resize((30, 30))
            im = OffsetImage(logo, zoom=1)
            ab = AnnotationBbox(im, (1.02, y), xycoords='axes fraction', box_alignment=(0, 0.5), frameon=False)
            ax.add_artist(ab)

        # Add team name with rank and record
        team_label = f"{i + 1}. {team.name} ({team.wins_count}-{team.losses_count})"
        ax.text(1.06, y, team_label, transform=ax.transAxes, va='center')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('weekly_top_25_rankings.png', dpi=300, bbox_inches='tight')
    print("Weekly rankings graph saved as 'weekly_top_25_rankings.png'")


def main(year):
    mov = True
    weekly_rankings = calculate_weekly_rankings(year, mov)
    plot_weekly_rankings(weekly_rankings)

if __name__ == "__main__":
    year = 2023
    main(year)