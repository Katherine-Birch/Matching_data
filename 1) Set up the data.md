1) Set up the data
    - Load the networks from SCmu.npy
    - standardise the edge weights to fixed range - this is useful for gradient stability across the architectures
    - Generate permutations from real data - this should be explicitly saved along with the seed integers
        - graph_id
        - permutation_seed
        - actual_permutation
        - inverse_permutation
    - Load also synthetic networks from the mecahnistic model 

2)  Benchmarking
    - use the FAQ model using scipy.optimize.quadratic_assignment(method='faq')
    - global topology using MLP-Sinkhorn model - this minimises Frobenius and/or L1 norm between the predicted soft-permitation recon and the original - differentiable outputs soft permutation matrix wehre rows and columns sum to 1
    - local embeddings using GNN and GAT - this uses the cross-entropy loss of the node similarity matrix - differentiable where the output is continuous node similarity matrix
    - Hungarian algorityhm applied identically across MLP and GNN to extract final hard assignments - this is the non-differentiable part where the output is binary so we know where each node goes
    
3) Complexity and permutation scaling
    - inference - we wrap forward and hungarian to time.time() blobk and record execution time per graph
    - we try scaling to new permutation limits
    - we output the permutation pool size, the alighnemt accuracy, adn the mean inference time and we can see the bottleneck as hungarian has complexity O(N^3)
    - We do additional experiemtns to see if there is a faster way :
        - Baseline - fixed set of permutations (save the seeds) - eg (0-9) 
        - Fine-tuning - fine-tune on a new set of seeds - eg (10-19)
        - Unseen - Evaluate the model on unseen permutations - eg (20-29)
            - On the baseline model
            - on the fine-tuned model
    - we track the Hungarian accuracy and L1 across the three (3.5) scenarios

4) synthetic data matching to biological data
    - This is difficult because 1) we dont know if an exact matching exists, 2) each network could be representing a different permutation
    - initially try the raw synthetic networks 
    - for any networks that do converge, isolate the hubs, and compare them to the empirical networks - probably none will converge properly though
    - calclate the number/percent of synthetic hubs that match to empirical hubs
