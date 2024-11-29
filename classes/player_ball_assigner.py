import sys

sys.path.append("../")
from utils import get_bbox_center


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_bbox_center(ball_bbox)

        for player_id, player in players.items():
            player_bbox = player["bbox"]
