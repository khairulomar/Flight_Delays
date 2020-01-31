def plot_all(df, num_of_plots_per_row = 6, target = None):
    
    import matplotlib.pyplot as plt
    from textwrap import wrap
    
    num_of_cols = df.shape[1]
    num_of_sets = num_of_cols // (num_of_plots_per_row + 1)
    k = 0
    for r, s in enumerate(range(num_of_sets)):
        color = 'blue' if (r % 2) == 0 else 'red'
        x1 = k
        x2 = k + num_of_plots_per_row
        k = k + num_of_plots_per_row + 1
        r = r * 2
        dfplot = df.iloc[:, x1:x2]
        cols_to_plot = list(dfplot.columns)

        fig, axes = plt.subplots(nrows=2, ncols=len(cols_to_plot), figsize=(18,4))
        for n, xcol in enumerate(cols_to_plot):
            axes[0,n].set_title("\n".join(wrap(xcol, 25)), fontsize=9)
            axes[0,n].scatter(dfplot[xcol], df[target], color=color, s=2)
            axes[0,n].set_ylabel(target, fontsize=8)
            axes[1,n].hist(dfplot[xcol], color=color)       
        plt.show()