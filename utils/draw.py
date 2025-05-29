import seaborn as sns
import matplotlib.pyplot as plt


##  violin
def violin(ax, data, x, y, order, palette, orient='v',
        hue=None, hue_order=None,
        mean_marker_size=6, err_capsize=.11, scatter_size=7):
        g_var = y if orient=='h' else x
        v_var = x if orient=='h' else y
        v=sns.violinplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=hue, hue_order=hue_order,
                            orient=orient, palette=palette, 
                            legend=False, alpha=.1, inner=None, density_norm='width',
                            ax=ax)
        plt.setp(v.collections, alpha=.35, edgecolor='none')
        sns.stripplot(data=data, 
                            x=x, y=y, order=order, 
                            hue=g_var if hue is None else hue, 
                            hue_order=order if hue is None else hue_order, 
                            orient=orient, palette=palette, 
                            size=scatter_size,
                            edgecolor='auto', jitter=True, alpha=.7,
                            dodge=False if hue is None else True,
                            legend=False, zorder=2,
                            ax=ax)
        sns.barplot(data=data, 
                            x=x, y=y, order=order, 
                            orient=orient, 
                            hue=hue, hue_order=hue_order,
                            errorbar='sd', linewidth=1, 
                            edgecolor=(0,0,0,0), facecolor=(0,0,0,0),
                            capsize=err_capsize, err_kws={'linewidth': 2.5,'color': [0.2, 0.2, 0.2]},
                            ax=ax)
        groupby = [g_var, hue] if hue is not None else [g_var]
        sns.stripplot(data=data.groupby(by=groupby)[v_var].mean().reset_index(), 
                        x=x, y=y, order=order, 
                        hue=hue, hue_order=hue_order, 
                        palette=[[.2]*3]*len(hue_order) if hue is not None else None,
                        dodge=False if hue is None else True,
                        marker='o', size=mean_marker_size, color=[.2]*3, ax=ax)
