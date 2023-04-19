def vis_pc(vis, img, title):
    vis.scatter(
        X=img.reshape(-1, 3),
        opts=dict(
            markersize=2,
            title=title,
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
        )
)