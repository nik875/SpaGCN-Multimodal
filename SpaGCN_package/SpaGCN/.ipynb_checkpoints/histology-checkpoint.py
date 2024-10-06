class HistologyEncoder:
    def __init__(self, image, size: int):
        """
        Initialize model, store dataset, and prepare to train on square areas of side length size.
        """
        self.image = image
        self.size = size

    def fit(self, spot_x, spot_y):
        """
        Given a list of x coordinates and a list of y coordinates for all the spots, extract all the
        square areas and assemble into a dataset. Then train an autoencoder to compress the
        information down to a single value, the third dimension of SpaGCN's 3D coordinates.
        """

    def predict(self, spot_x, spot_y):
        """
        Given the spot coordinates, extract square areas and predict the bottleneck representations.
        """

    def fit_predict(self, spot_x, spot_y):
        self.fit(spot_x, spot_y)
        return self.predict(spot_x, spot_y)
