
import fiona
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('../../data/trackdf.csv')

    # define schema
    schema = {
        'geometry': 'LineString',
        'properties': [('trackid', 'int'), ('cluster', 'int')]
    }

    # open a fiona object
    lineShp = fiona.open('../../data/proposed_tracks.shp', mode='w', driver='ESRI Shapefile',
                         schema=schema, crs="EPSG:4326")

    tracks = df.trackid.unique()

    for track in tracks:
        dft = df[df.trackid == track]
        xylist = []
        rowName = track

        for i, row in dft.iterrows():
            cluster = row["cluster"]
            xylist.append((row["longitude"], row["latitude"]))
        # save record and close shapefile
        rowDict = {
            'geometry': {'type': 'LineString',
                         'coordinates': xylist},
            'properties': {'trackid': int(track),
                           'cluster': int(cluster)},
        }
        lineShp.write(rowDict)

    lineShp.close()