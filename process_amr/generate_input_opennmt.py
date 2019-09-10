"""
Leonardo Ribeiro
ribeiro@aiphes.tu-darmstadt.de
"""
import sys
import getopt
import networkx as nx

if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")


def process_data_bipartite(nodes, edges):
    bipartite_nodes = nodes.split()
    bipartite_edges = []

    original_nodes = []
    original_label_edges = []
    map_nodes = {}
    map_edges = {}
    original_edges_source = []
    original_edges_target = []


    for idx, n in enumerate(bipartite_nodes):
        if not n.startswith(':'):
            original_nodes.append(n)
            map_nodes[idx] = len(original_nodes) - 1
        else:
            original_label_edges.append(n)
            map_edges[idx] = len(original_label_edges) - 1
            original_edges_source.append(-1)
            original_edges_target.append(-1)

    for e in edges.split():
        e = e.replace('(', '').replace(')', '')
        if e[-1] == 'd' or e[-1] == 's':
            e = e.split(',')
            e[0] = int(e[0])
            e[1] = int(e[1])

            bipartite_edges.append((e[0], e[1]))

    if len(bipartite_nodes) == 1 and len(bipartite_edges) == 0:
        bipartite_edges.append((0, 0))

    return bipartite_nodes, bipartite_edges


def create_graph_bipartite(bipartite_nodes, bipartite_edges):
    g = nx.DiGraph()
    for e in bipartite_edges:
        g.add_edge(e[0], e[1])

    return g


def read_dataset(file_nodes, file_edges, file_surfaces, part):
    print(file_nodes)
    with open(file_nodes, 'r', encoding="utf-8") as dataset_file:
        dataset_nodes = dataset_file.readlines()
    with open(file_edges, 'r', encoding="utf-8") as dataset_file:
        dataset_edges = dataset_file.readlines()
    with open(file_surfaces, 'r', encoding="utf-8") as dataset_file:
        dataset_surfaces = dataset_file.readlines()

    all_instances = []

    for idx, (nodes, edges, surfaces) in enumerate(zip(dataset_nodes, dataset_edges, dataset_surfaces)):

        surfaces = surfaces.strip().split()
        nodes, edges = process_data_bipartite(nodes, edges)
        all_instances.append((nodes, edges, surfaces))

    print('Total of {} instances processed in {} dataset'.format(len(dataset_nodes), part))

    return all_instances


def build_graph_bipartite(instance):

    nodes = instance[0]
    edges = instance[1]

    src_nodes = nodes
    src_edges_node1 = []
    src_edges_node2 = []

    for j, edge in enumerate(edges):

        # graph with nodes and edges rep
        src_edges_node2.append(edge[0])
        src_edges_node1.append(edge[1])

    src_edges_node1 = list(map(str, src_edges_node1))
    src_edges_node2 = list(map(str, src_edges_node2))

    assert len(src_edges_node1) == len(src_edges_node2)

    return " ".join(src_nodes), (" ".join(src_edges_node1), " ".join(src_edges_node2))


def create_files_bipartite(data, part, path):

    source_nodes_out = []
    source_edges_out_node1 = []
    source_edges_out_node2 = []
    sents = []

    for _id, instance in enumerate(data):

        source_nodes, source_edges = build_graph_bipartite(instance)
        source_nodes_out.append(source_nodes)
        source_edges_out_node1.append(source_edges[0])
        source_edges_out_node2.append(source_edges[1])
        sents.append(' '.join(instance[2]))

    with open(path + '/' + part + '-amr-nodes.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(source_nodes_out))
    with open(path + '/' + part + '-amr-node1.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(source_edges_out_node1))
    with open(path + '/' + part + '-amr-node2.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(source_edges_out_node2))
    with open(path + '/' + part + '-amr-tgt.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(sents))


def input_files(path):

    parts = ['train', 'dev', 'test']

    for part in parts:
        file_nodes = path + '/' + part + '/nodes.pp.txt'
        file_edges = path + '/' + part + '/triples.pp.txt'
        file_surfaces = path + '/' + part + '/surface.pp.txt'

        print('Processing AMR files for bipartite graph...')
        data = read_dataset(file_nodes, file_edges, file_surfaces, part)
        print('Done')

        print('Creating opennmt files...')
        create_files_bipartite(data, part, path)
        print('Done')

    print('Files necessary for training/evaluating/test are written on disc.')


def main(argv):
    usage = 'usage:\npython generate_input_opennmt.py -i <data-directory>' \
           '\ndata-directory is the directory where AMR json files are located.'
    try:
        opts, args = getopt.getopt(argv, 'i:', ['inputdir='])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    input_data = False
    for opt, arg in opts:
        if opt in ('-i', '--inputdir'):
            inputdir = arg
            input_data = True
        else:
            print(usage)
            sys.exit()
    if not input_data:
        print(usage)
        sys.exit(2)
    print('Input directory is {}'.format(inputdir))

    input_files(inputdir)


if __name__ == "__main__":
    main(sys.argv[1:])
