mkdir ../results/run_results

# Run predictions for protein coding genes
# Iteration 0
python pc_lncrna_dnn.py --input_data_file ../results/pan_cancer_data/pc_lncrna_ge_data_all.npz --dnn_model ../results/pan_cancer_data/models/pc_models/Iter0_pc_normal_vs_cancer_dnn --svm_model ../results/pan_cancer_data/models/pc_models/Iter0_pc_normal_vs_cancer_svm.joblib --rf_model ../results/pan_cancer_data/models/pc_models/Iter0_pc_normal_vs_cancer_rf.joblib --selected_gene_class protein_coding_genes --output_file ../results/run_results/Iter0_pc_normal_vs_cancer

# Iteration 1
python pc_lncrna_dnn.py --input_data_file ../results/pan_cancer_data/pc_lncrna_ge_data_all.npz --dnn_model ../results/pan_cancer_data/models/pc_models/Iter1_pc_normal_vs_cancer_dnn --svm_model ../results/pan_cancer_data/models/pc_models/Iter1_pc_normal_vs_cancer_svm.joblib --rf_model ../results/pan_cancer_data/models/pc_models/Iter1_pc_normal_vs_cancer_rf.joblib --selected_gene_class protein_coding_genes --output_file ../results/run_results/Iter1_pc_normal_vs_cancer

# Iteration 2
python pc_lncrna_dnn.py --input_data_file ../results/pan_cancer_data/pc_lncrna_ge_data_all.npz --dnn_model ../results/pan_cancer_data/models/pc_models/Iter2_pc_normal_vs_cancer_dnn --svm_model ../results/pan_cancer_data/models/pc_models/Iter2_pc_normal_vs_cancer_svm.joblib --rf_model ../results/pan_cancer_data/models/pc_models/Iter2_pc_normal_vs_cancer_rf.joblib --selected_gene_class protein_coding_genes --output_file ../results/run_results/Iter2_pc_normal_vs_cancer

# Run predictions for lncRNA genes
# Iteration 0
python pc_lncrna_dnn.py --input_data_file ../results/pan_cancer_data/pc_lncrna_ge_data_all.npz --dnn_model ../results/pan_cancer_data/models/lncrna_models/Iter0_lncrna_normal_vs_cancer_dnn --svm_model ../results/pan_cancer_data/models/lncrna_models/Iter0_lncrna_normal_vs_cancer_svm.joblib --rf_model ../results/pan_cancer_data/models/lncrna_models/Iter0_lncrna_normal_vs_cancer_rf.joblib --selected_gene_class lncRNA_genes --output_file ../results/run_results/Iter0_lncrna_normal_vs_cancer

# Iteration 1
python pc_lncrna_dnn.py --input_data_file ../results/pan_cancer_data/pc_lncrna_ge_data_all.npz --dnn_model ../results/pan_cancer_data/models/lncrna_models/Iter1_lncrna_normal_vs_cancer_dnn --svm_model ../results/pan_cancer_data/models/lncrna_models/Iter1_lncrna_normal_vs_cancer_svm.joblib --rf_model ../results/pan_cancer_data/models/lncrna_models/Iter1_lncrna_normal_vs_cancer_rf.joblib --selected_gene_class lncRNA_genes --output_file ../results/run_results/Iter1_lncrna_normal_vs_cancer

# Iteration 2
python pc_lncrna_dnn.py --input_data_file ../results/pan_cancer_data/pc_lncrna_ge_data_all.npz --dnn_model ../results/pan_cancer_data/models/lncrna_models/Iter2_lncrna_normal_vs_cancer_dnn --svm_model ../results/pan_cancer_data/models/lncrna_models/Iter2_lncrna_normal_vs_cancer_svm.joblib --rf_model ../results/pan_cancer_data/models/lncrna_models/Iter2_lncrna_normal_vs_cancer_rf.joblib --selected_gene_class lncRNA_genes --output_file ../results/run_results/Iter2_lncrna_normal_vs_cancer

# Run predictions for splicing
#Iteration 0
python splicing_dnn.py --input_data_file ../results/pan_cancer_data/splicing_data.npz --dnn_model ../results/pan_cancer_data/models/splicing_models/Iter0_splicing_normal_vs_cancer_dnn --svm_model ../results/pan_cancer_data/models/splicing_models/Iter0_splicing_normal_vs_cancer_svm.joblib --rf_model ../results/pan_cancer_data/models/splicing_models/Iter0_splicing_normal_vs_cancer_rf.joblib --output_file ../results/run_results/Iter0_splicing_normal_vs_cancer

#Iteration 1
python splicing_dnn.py --input_data_file ../results/pan_cancer_data/splicing_data.npz --dnn_model ../results/pan_cancer_data/models/splicing_models/Iter1_splicing_normal_vs_cancer_dnn --svm_model ../results/pan_cancer_data/models/splicing_models/Iter1_splicing_normal_vs_cancer_svm.joblib --rf_model ../results/pan_cancer_data/models/splicing_models/Iter2_splicing_normal_vs_cancer_rf.joblib --output_file ../results/run_results/Iter1_splicing_normal_vs_cancer

#Iteration 2
python splicing_dnn.py --input_data_file ../results/pan_cancer_data/splicing_data.npz --dnn_model ../results/pan_cancer_data/models/splicing_models/Iter2_splicing_normal_vs_cancer_dnn --svm_model ../results/pan_cancer_data/models/splicing_models/Iter2_splicing_normal_vs_cancer_svm.joblib --rf_model ../results/pan_cancer_data/models/splicing_models/Iter2_splicing_normal_vs_cancer_rf.joblib --output_file ../results/run_results/Iter2_splicing_normal_vs_cancer


