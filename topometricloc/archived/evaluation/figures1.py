import os.path as path
import matplotlib.pyplot as plt
import pandas as pd

map_path = '/Users/ceebskent/Downloads/map-2.png'
rtk_dir = '/Users/ceebskent/Downloads/rtk/'
traverses_date = ['2015-05-19-14-06-38',
                  '2014-11-21-16-07-03', '2014-12-16-18-44-24',
                  '2015-07-29-13-09-26', '2015-04-24-08-15-07']
traverses = ['Overcast', 'Dusk', 'Night', 'Rain', 'Sun']
lims = {"Sun": (1429864457133615, 1429865645036369),
        "Rain": (1438176485086715, 1438177255233697)}
intervals = [0., 0.00010, 0.00020, 0.00030, 0.00040]
intervals1 = [0., 0.00010, 0.00020, 0.00023, 0.00026]

lat_lims = [51.75, 51.764]
lon_lims = [-1.269, -1.251]
BBox = [*lon_lims, *lat_lims]

dfs = {}
for traverse, date in zip(traverses, traverses_date):
    dfs[traverse] = pd.read_csv(path.join(rtk_dir, date, 'rtk.csv'))

img = plt.imread(map_path)
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(*lon_lims)
ax.set_ylim(*lat_lims)
for i, (traverse, df) in enumerate(dfs.items()):
    if traverse in lims:
        mask = df['timestamp'].between(lims[traverse][0], lims[traverse][1])
        df = df[mask]
    ax.scatter(df.longitude - intervals[i], df.latitude + intervals1[i], zorder=1,
               alpha= 0.5, s=20, label=traverses[i])
ax.imshow(img, zorder=0, extent=BBox, aspect='equal')
ax.legend(fontsize=16)
ax.axis("off")
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
plt.show()
