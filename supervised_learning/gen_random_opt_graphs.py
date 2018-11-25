import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import networkx as nx


nn_root_path = 'C:\\Users\\danie\\Desktop\\School\\2018_fall\\machine_learning\\randomized_optimization\\nn_results\\'
problem_root_path = 'C:\\Users\\danie\\Desktop\\School\\2018_fall\\machine_learning\\randomized_optimization\\problem_results\\'

def generate_nn_plot(path, input_file, title, ref_line):
    results = pd.read_csv(path + input_file + ".csv", header=0)
    results = results.set_index(list(results)[0])

    sns.set(style="darkgrid")

    ax = sns.lineplot(data=results)
    ax.set_title(title)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("iteration")

    ax.axhline(ref_line, linewidth=2, ls=':', color='r', clip_on=True)

    fig = ax.get_figure()
    fig.savefig(path + input_file + '.png')

    plt.show()

def generate_max_k_problem_plot(path, input_file, y_label, title):
    results = pd.read_csv(path + input_file + ".csv", header=0)
    results = results.set_index(list(results)[0])

    sns.set(style="darkgrid")

    ax = sns.lineplot(data=results)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("K")

    fig = ax.get_figure()
    fig.savefig(path + input_file + '.png')

    plt.show()

def generate_queens_problem_plot(path, input_file, y_label, title):
    results = pd.read_csv(path + input_file + ".csv", header=0)
    results = results.set_index(list(results)[0])

    sns.set(style="darkgrid")

    ax = sns.lineplot(data=results)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("# of queens")

    fig = ax.get_figure()
    fig.savefig(path + input_file + '.png')

    plt.show()

def generate_tsp_map(path, input_file):
    results = pd.read_csv(path + input_file + ".csv", header=0)

    nodeList = pd.unique(results[['to', 'from']].values.ravel('K'))

    graph = nx.Graph()
    graph.add_nodes_from(nodeList)

    for index, row in results.iterrows():
        graph.add_edge(row['to'], row['from'], weight=row['weight'] * 2, color='gray')

    #nx.draw(graph, with_labels=True)
    #plt.show()

    return graph

def generate_tsp_paths(path, network_map_file_name, file_name):
    results = pd.read_csv(path + file_name + ".csv", header=None)
    results = results.set_index(list(results)[0])

    colors = {'random hill climber' : 'red',
              'simulated annealing' : 'purple',
              'genetic algorithm'   : 'green',
              'mimic'               : 'blue'}

    for index, row in results.iterrows():
        graph = generate_tsp_map(path, network_map_file_name)
        generate_path(path, graph, row, index, colors[index])

def generate_path(path, graph, row, name, color):
    frame = pd.DataFrame(row)
    originator = 0
    for index, row in frame.iterrows():
        node = int(row[0])
        graph.edges[originator, node]['color'] = color
        originator = node

    edge_colors = [e[2]['color'] for e in graph.edges(data=True)]
    edge_widths = [e[2]['weight'] for e in graph.edges(data=True)]

    plt.figure()
    plt.title(name + ' path')

    nx.draw(graph, with_labels=True, edge_color=edge_colors, width=edge_widths, node_size=5)

    plt.savefig(path + name + "_path.png", format="PNG")
    plt.show()

def generate_tsp_problem_opts(path, input_file, y_label, title):
    results = pd.read_csv(path + input_file + ".csv", header=0)
    results = results.set_index(list(results)[0])

    sns.set(style="darkgrid")

    ax = sns.lineplot(data=results)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("nodes")

    fig = ax.get_figure()
    fig.savefig(path + input_file + '.png')

    plt.show()

def get_averages(path, input_file):
    results = pd.read_csv(path + input_file + ".csv", header=0)
    results = results.set_index(list(results)[0])


    pass

#generate_nn_plot(nn_root_path, "random_results_encoded_data_unacceptable", "random optimization of unacceptable cars", 0.9947)
#generate_nn_plot(nn_root_path, "random_results_encoded_data_acceptable", "random optimization of acceptable cars", 0.9719)
#generate_nn_plot(nn_root_path, "random_results_encoded_data_good", "random optimization of good cars", 0.9824)
#generate_nn_plot(nn_root_path, "random_results_encoded_data_very_good", "random optimization of very good cars", 0.9824)
#generate_nn_plot(nn_root_path, "annealing_results_encoded_data_very_good", "random optimization of very good cars", 0.9824)

#generate_max_k_problem_plot(problem_root_path, "max_k_color_optimals", "solved at iteration", "Max K Color.  Iterations / K")
#generate_max_k_problem_plot(problem_root_path, "max_k_color_times", "milliseconds", "Max K Color.  Time (ms) / K")

#generate_queens_problem_plot(problem_root_path, "n_queens_color_times", "milliseconds", "Queens: Time (ms) / Number of Queens")

#graph = generate_tsp_map(problem_root_path, "tsp_map")
#genereate_tsp_path(problem_root_path, graph, "tsp_paths")

#generate_tsp_paths(problem_root_path, "tsp_map", "tsp_paths")

#generate_tsp_problem_opts(problem_root_path, "tsp_optimals", "distance", "Travelling Salesman:  distance / nodes")
#generate_tsp_problem_opts(problem_root_path, "tsp_times", "milliseconds", "Travelling Salesman:  time / nodes")

get_averages(problem_root_path, "tsp_times")