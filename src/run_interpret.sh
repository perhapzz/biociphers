mkdir ../results/run_results

# Run interpretation for protein coding genes, be careful running this, this might take a few hours to run without a GPU
python interpret_pc_lncrna_models.py --input_data_file ../results/pan_cancer_data/pc_lncrna_ge_data_all.npz --dnn_model ../results/pan_cancer_data/models/pc_models/Interpret_pc_normal_vs_cancer_dnn --selected_gene_class protein_coding_genes --output_path ../results/run_results/ 

# Run interpretation for lncRNA genes, be careful running this, this might take a few hours to run without a GPU
python interpret_pc_lncrna_models.py --input_data_file ../results/pan_cancer_data/pc_lncrna_ge_data_all.npz --dnn_model ../results/pan_cancer_data/models/lncrna_models/Interpret_lncrna_normal_vs_cancer_dnn --selected_gene_class lncRNA_genes --output_path ../results/run_results/

# Run interpretation for lncRNA genes, be careful running this, this might take a few hours to run without a GPU
python interpret_splicing_models.py --input_data_file ../results/pan_cancer_data/splicing_data.npz --dnn_model ../results/pan_cancer_data/models/splicing_models/Interpret_splicing_normal_vs_cancer_dnn --output_path ../results/run_results/


