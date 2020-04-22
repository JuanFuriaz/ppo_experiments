import argparse
import matplotlib.pyplot as plt
import pandas as pd
import glob
plt.style.use("seaborn-colorblind")


class ModelStruct(object):

    def __init__(self, name, pattern):
        self.name = name
        self.pattern = pattern
        self.stats = {}
        self.df = None

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.name, self.pattern)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='logdir that is recursively searched for csv files')
    args = parser.parse_args()

    csv_files = glob.glob("%s/**/*.csv" % args.logdir, recursive=True)
    models = [ModelStruct("VAE", "vae"), ModelStruct("InfoMax", "infomax"), ModelStruct("Raw Pixel", "rawpixel")]
    columns = ["episode", "score_mean", "score_median", "score_std", "score_min", "score_max", "score_moving_mean"]
    stat = "score_median"

    # collect statistics from csv files
    for m in models:
        csv_files_filtered = [f for f in csv_files if m.pattern in f]
        if len(csv_files_filtered) == 0:
            raise ValueError("No csv files for model %s" % m)
        df = pd.DataFrame(columns=columns)
        for f in csv_files_filtered:
            tmp = pd.read_csv(f, names=columns)
            df = pd.concat([df, tmp], axis=0)
        m.df = df

    # plot stats across episodes
    plt.figure(dpi=300)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for m, c in zip(models, colors):
        means = m.df.groupby("episode")[stat].mean()
        stds = m.df.groupby("episode")[stat].std()
        plt.plot(means.index, means, color=c, linewidth=1.5, label=m.name)
        plt.fill_between(means.index, means - stds, means + stds, alpha=.2, color=c, linewidth=0.5)
        # plt.errorbar(means.index, means, stds, color=c, alpha=0.2)
    plt.legend(loc="upper left")
    plt.savefig("summary.png")
    plt.show()
