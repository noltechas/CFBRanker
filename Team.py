class Team:
    def __init__(self, id, name, conference, division):
        self.id = id
        self.name = name
        self.conference = conference
        self.division = division
        self.wins = {}  # Will store {opponent_id: [(count, margin), ...]}
        self.losses = {}  # Will store {opponent_id: [(count, margin), ...]}
        self.wins_count = 0
        self.fbs_wins_count = 0
        self.losses_count = 0

    def add_win(self, opponent_id, is_fbs, margin):
        if opponent_id not in self.wins:
            self.wins[opponent_id] = []
        self.wins[opponent_id].append((1, margin))
        self.wins_count += 1
        if is_fbs:
            self.fbs_wins_count += 1

    def add_loss(self, opponent_id, margin):
        if opponent_id not in self.losses:
            self.losses[opponent_id] = []
        self.losses[opponent_id].append((1, margin))
        self.losses_count += 1

    def __str__(self):
        return f"{self.name} ({self.wins_count}-{self.losses_count})"