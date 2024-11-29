from sklearn.cluster import KMeans


class TeamAssigner:
    # The development of this class was made in the 'color_assignement.ipynb' notebook
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_cluster_model(self, image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters, we use 'k-means++' because it's the fastest way
        # if we increase 'n_init' parameter the model will be more confident
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Get the color of the player jerseys from a given frame and player bbox

        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2]), :]

        # Take the top half of the image, because is mostly where the colors of the jersey will be
        top_half_image = image[0 : int(image.shape[0] / 2), :]

        # Init the cluster model to cluster the top half of the image
        kmeans = self.get_cluster_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the original image shape
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )

        # Get the player cluster and color
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        bg_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - bg_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):

        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # this time we cluster the array of colors we just created
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # We create this condition so if we already have a team for the specific
        # 'player_id' we don't run all the clustering model and algorithm again
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        # Hard code the goalkeepers
        if player_id == 81:
            team_id = 1
        if player_id == 20 | 193:
            team_id = 2

        self.player_team_dict[team_id] = team_id

        return team_id
