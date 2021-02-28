from infomap import Infomap
import numpy as np
from info_map_helpers import expand_labels
import matplotlib.pyplot as plt
from skimage.future.graph import RAG as rag
from skimage.measure import label
import BuildAllenTree as BAT


class Cellograph():
    def __init__(self, base = '/bigstore/GeneralStorage/Evan/InfoMap/Cell_Type_Region_Conditional_Information/'):
        # load necessary files
        print('Loading necessary files...')
        self.cell_seg_map = np.load(base+'cell_seg_map_v2.npy')
        cell_types = open(base+'celltypes_v2.txt', 'r')
        cell_types = cell_types.readlines()
        self.cell_types = [i.split('\n')[0] for i in cell_types]  # this is the raw cell types at bottom of dendogram (132 unique types)
        self.dendogram_levels = np.load(base+'dendogram_levels.npy')
        
        self.expanded_im = self._expand_map()
        
        print('Building graph...')
        # use a region adjacency graph (rag) on expanded labels to detect
        self.exp_rag = rag(self.expanded_im)
        self.cell_edges = [edge for edge in list(self.exp_rag.edges) if -1 not in edge] # -1 is the background
        self.cell_nodes = list(self.exp_rag.nodes)[1:]


    def _expand_map(self, start_ix=8):
        # make graph and edges
        # keep expanding labels until it's one contiguous block
        for i in range(start_ix,100):  # I know it's 8  now...
            expanded = expand_labels(self.cell_seg_map+1, i)  # need the +1 to
            if np.max(label(expanded>0))==1:
                print('Expanded segmentation pixel thickness:', i)
                break
            else:
                print('not', i)
        expanded -= 1  # makes it equivalent tot cell_seg_map again
        return expanded



    def make_cell_type_image(self, cell_type_list, seg_map_img):
        cts = np.array(list(set(cell_type_list)))
        ct_img = np.zeros_like(seg_map_img)-1
        for i, ct in enumerate(cell_type_list):
            if not i%2000:
                print(i)
            coords = seg_map_img==self.cell_nodes[i]
            ct_img[coords] = np.where(cts==ct)[0][0]

        return ct_img

    
    def run_rigid_dendogram_sweep_expt(self):
        self.results_rds = {i:{} for i in range(len(self.dendogram_levels))}

        for d_ix, level in enumerate(self.dendogram_levels):
            print('Dendogram Level:', level)
            ctcollapse = BAT.get_celltype_name_map(BAT.buildallentree(level,False,False), 
                                                     BAT.buildallentree(0,False,False))
            cell_types_combined = [ctcollapse[i] for i in self.cell_types]
            unq_cts = list(set(cell_types_combined))
            cell_types_dict = {ct: unq_cts.index(ct) for ct in cell_types_combined}

            # label/color every cell by its type!
            self.results_rds[d_ix]['ct img'] = self.make_cell_type_image(cell_types_combined, self.cell_seg_map[:,:,0])

            edge_weights = {edge:1 for edge in self.cell_edges.copy()}
            new_edges = [tuple(i) for i in np.array(self.cell_edges.copy())]
            for cell in range(len(cell_types_combined)):  # go through each cell and combine all its neighbors that are the same type
                if not cell%2000:
                    print('Cell:', cell)
                # this group has the same cell type as the current cell
                ct_group = [c for i, c in enumerate(self.cell_nodes) if cell_types_dict[cell_types_combined[i]]==
                                                                        cell_types_dict[cell_types_combined[cell]]]

                # some edges need to be deleted and others need to be changed
                # edges with both current cell and any cell with the same cell type should be deleted (functionally merging nodes)
                # edges with any ids that were merged should be changed to the current cell's id
                change_nodes = []
                delete_edges = []
                for edge in new_edges:
                    # remove edge if its got our current cell and shares the cell type (is in ct_group)
                    if self.cell_nodes[cell] in edge and edge[0] in ct_group and edge[1] in ct_group: # remove the edge
                        # add weight to distributed at the end to remaining nodes
                        change_nodes.append([i for i in edge if i!=self.cell_nodes[cell]][0])
                        delete_edges.append(edge)
                        delete_edges.append((edge[1], edge[0]))
                    else:  # don't change the edge
                        pass

                new_edges = [e for e in new_edges if e not in delete_edges]
                edge_weights = {edge: edge_weights[edge] for edge in edge_weights.keys() if edge not in delete_edges} 

                # replace all nodes in change_nodes with the current cell's node
                for e_ix, edge in enumerate(new_edges):
                    if edge[0] in change_nodes and edge[1] not in change_nodes:
                        new_edge = (self.cell_nodes[cell], edge[1])
                        new_edges[e_ix] = new_edge  # replace edge with new_edge
                        for ej, e in enumerate(new_edges):  # really be sure it is replaced
                            if e==edge:
                                new_edges[ej] = new_edge
                        if edge in edge_weights.keys():
                            edge_weights[new_edge] = edge_weights[edge]  # copy old edge weight to new edge
                        edge_weights.pop(edge, None)  # remove old edge

                    elif edge[1] in change_nodes and edge[0] not in change_nodes:
                        new_edge = (edge[0], self.cell_nodes[cell])
                        new_edges[e_ix] = new_edge  # replace edge with new_edge
                        for ej, e in enumerate(new_edges):  # really be sure it is replaced
                            if e==edge:
                                new_edges[ej] = new_edge
                        if edge in edge_weights.keys():
                            edge_weights[new_edge] = edge_weights[edge]  # copy old edge weight to new edge
                        edge_weights.pop(edge, None)  # remove old edge

                    elif edge[0] in change_nodes and edge[1] in change_nodes:  # if both are to be changed, I delete? double check
                        new_edges.remove(edge)
                        edge_weights.pop(edge, None)  # remove old edge

                # distribute weights to remaining edges involving our cell's node, if not in change edges
                for edge in new_edges:
                    if self.cell_nodes[cell] in edge:
                        # 1st need to figure out how many edges are connected to this cells node to calculate reweight factor (rw)
                        rw = 1/len([edge for edge in new_edges if self.cell_nodes[cell] in edge])
                        if edge in list(edge_weights.keys()):
                            edge_weights[edge] += rw
                        elif (edge[1], edge[0]) in list(edge_weights.keys()):
                            edge_weights[(edge[1], edge[0])] += rw
                        else:  # copy the most common value from the edges connected to our node
                            edge_weights[edge] = np.random.choice([edge_weights[e] for e in new_edges 
                                                                   if self.cell_nodes[cell] in e and 
                                                                   e in edge_weights.keys()], 1)
            self.results_rds[d_ix]['edges'] = new_edges
            self.results_rds[d_ix]['edge_weights'] = edge_weights  
            self.run_infomap(self.results_rds, d_ix)
            
        print('Experiment complete. Results saved in self.results_rds')
    
    
    def run_infomap(self, results, level, infomap_args='--undirected'):
        # run infomap on the edges
        results[level]['codelen'] = []
        imap = Infomap(infomap_args)  
        for link, weight in results[level]['edge_weights'].items():
            imap.add_link(link[0], link[1], weight=weight)  # only way I could find to add the weights
        imap.run()
        results[level]['codelen'].append(imap.codelength)