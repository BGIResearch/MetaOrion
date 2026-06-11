import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
from matplotlib.patches import Patch

# 忽略 geopandas 警告
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['figure.dpi'] = 800
plt.rcParams['savefig.dpi'] = 800
plt.rc('font', size=32)

# ================= 1. 读取并处理数据 =================
file_path = "pivot_bodysite.xlsx"
df = pd.read_excel(file_path)

df = df[df['country'] != 'Total'].copy()

categories = ['Body_fluids', 'Gut', 'Nasal', 'Oral', 'Skin', 'Unine', 'Vagina']
for cat in categories:
    if cat in df.columns:
        df[cat] = pd.to_numeric(df[cat], errors='coerce').fillna(0)
if 'Total' in df.columns:
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)

# ================= 2. 加载世界地图数据 =================
world_path = r"ne_110m_admin_0_countries.geojson.txt"
world = gpd.read_file(world_path)

col_name = 'ISO_A3_EH' if 'ISO_A3_EH' in world.columns else 'ISO_A3'

for name_col in ['NAME', 'ADMIN', 'name', 'admin']:
    if name_col in world.columns:
        mask = world[name_col].isin(['Taiwan', 'Hong Kong', 'Macao', 'Macau'])
        world.loc[mask, col_name] = 'CHN'

for name_col in ['NAME', 'ADMIN', 'name', 'admin']:
    if name_col in world.columns:
        sg_mask = world[name_col].str.contains('Singapore', case=False, na=False) | \
                  world[name_col].isin(['新加坡', 'SGP'])
        world.loc[sg_mask, col_name] = 'SGP'

sg_geometry = gpd.points_from_xy([103.8198], [1.3521]).buffer(0.5)
sg_row = gpd.GeoDataFrame({
    col_name: ['SGP'],
    'NAME': ['Singapore'],
    'CONTINENT': ['Asia'],
    'geometry': sg_geometry
}, crs=world.crs)

if not world[world[col_name] == 'SGP'].any().any():
    world = pd.concat([world, sg_row], ignore_index=True)

merged = world.merge(df, left_on=col_name, right_on='country', how='left')

# ================= 3. 初始化画布与底图 =================
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
world.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)

# ================= 4. 绘制热力图 =================
data_to_plot = merged[merged['Total'] > 0]

if not data_to_plot.empty:
    vmin = data_to_plot['Total'].min()
    vmax = data_to_plot['Total'].max()

    cax = ax.inset_axes([0.08, 0.15, 0.015, 0.5])

    data_to_plot.plot(
        column='Total', ax=ax, cmap='Blues', edgecolor='black', linewidth=0.5,
        norm=colors.LogNorm(vmin=max(vmin, 1), vmax=vmax),
        legend=True,
        cax=cax,
        legend_kwds={
            'label': "Total Samples (Log Scale)",
            'orientation': "vertical"
        }
    )

    cax.yaxis.set_label_position('left')
    cax.yaxis.set_ticks_position('left')

# ================= 5. 在各大洲上叠加饼图 =================
categories = sorted(categories)

category_colors = {
    "Gut": '#F1DFA4',
    "Oral": '#F5AA61',
    "Vagina": '#9DD0C7',
    "Skin": '#E58579',
    "Body_fluids": '#D9BDDB',
    "Nasal": '#8AB1D2',
    "Unine": '#B4D2C3'
}

continent_col = None
for col in ['CONTINENT', 'continent', 'REGION_UN']:
    if col in merged.columns:
        continent_col = col
        break

if continent_col:
    continent_data = merged.groupby(continent_col)[categories].sum()
    continent_data['Continent_Total'] = continent_data[categories].sum(axis=1)
    continents_geo = world.dissolve(by=continent_col)

    manual_centers = {
        'North America': (-100, 45),
        'Europe': (15, 50),
        'Asia': (90, 40),
        'Africa': (20, 0),
        'South America': (-60, -15),
        'Oceania': (140, -25)
    }

    min_pie_size = 10
    max_pie_size = 40
    max_total = continent_data['Continent_Total'].max() if not continent_data.empty else 1

    for continent, row in continents_geo.iterrows():
        if continent not in continent_data.index:
            continue

        site_counts = continent_data.loc[continent]
        valid_sites = site_counts[categories][site_counts[categories] > 0]
        continent_total = site_counts['Continent_Total']

        if len(valid_sites) > 0 and continent_total > 0:
            if continent in manual_centers:
                x, y = manual_centers[continent]
            else:
                centroid = row['geometry'].centroid
                x, y = centroid.x, centroid.y

            pie_size = (continent_total / max_total) * (max_pie_size - min_pie_size) + min_pie_size

            ax_pie = ax.inset_axes(
                [x - pie_size / 2, y - pie_size / 2, pie_size, pie_size],
                transform=ax.transData,
                zorder=10
            )

            colors_for_pie = [category_colors[cat] for cat in valid_sites.index]

            ax_pie.pie(
                valid_sites,
                colors=colors_for_pie,
                wedgeprops=dict(edgecolor='black', linewidth=0.5)
            )
            ax_pie.set_aspect('equal')
            ax_pie.patch.set_alpha(0)

# ================= 6. 添加图例与输出 =================
legend_elements = [Patch(facecolor=category_colors[cat], edgecolor='none', label=cat) for cat in categories]

ax.legend(
    handles=legend_elements,
    loc='center left',
    frameon=False,
    handletextpad=0.3,
    bbox_to_anchor=(0.09, 0.34) # 固定在 x=0.04, y=0.05 的位置，彻底与上面的比例尺错开
)

plt.axis('off')
plt.tight_layout()
plt.savefig('World_Map_continent_Pie.pdf', dpi=800, bbox_inches='tight', format='pdf')