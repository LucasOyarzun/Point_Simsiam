def vis_pc(vis, pc, title):
    vis.scatter(
        X=pc,
        opts=dict(
            markersize=2,
            title=title,
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
        )
)
