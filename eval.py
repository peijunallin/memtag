# import os
# import argparse
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import logging
# import glob

# # Set up logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
# logger = logging.getLogger(__name__)


import os
import argparse
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import logging
import glob
from nltk.stem import PorterStemmer
from collections import Counter
import regex
import string
from rouge import Rouge

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Initialize Porter Stemmer
ps = PorterStemmer()

# Text normalization function
def normalize_answer(s):
    s = s.replace(',', "")
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# F1 score functions
def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])

# Exact match functions
def exact_match_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return set(prediction.split()) == set(ground_truth.split())

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

# ROUGE-L score functions
def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    prediction = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
    ground_truth = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]

def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])




def evaluate_predictions(predictions_file, metrics=['f1', 'em']):
    """
    Evaluate predictions against ground truth answers using F1 and Exact Match metrics.
    
    Args:
        predictions_file: Path to the CSV file with predictions
        metrics: List of metrics to compute
    
    Returns:
        Dictionary with evaluation results
    """
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Ensure required columns exist
    required_columns = ['question', 'answer', 'predictions']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {predictions_file}")
    
    # Filter out rows with BLANK predictions
    df_evaluated = df[df['predictions'] != 'BLANK'].copy()
    
    if len(df_evaluated) == 0:
        logger.warning(f"No valid predictions found in {predictions_file}")
        return {
            'file': predictions_file,
            'total_questions': len(df),
            'evaluated_questions': 0,
            'f1_score': 0.0,
            'exact_match': 0.0
        }
    
    # Calculate metrics
    f1_scores = []
    em_scores = []
    
    for _, row in tqdm(df_evaluated.iterrows(), total=len(df_evaluated), desc="Evaluating"):
        prediction = str(row['predictions'])
        ground_truth = str(row['answer'])
        
        # Handle multiple ground truths (if they're comma-separated)
        ground_truths = [g.strip() for g in ground_truth.split(',')]
        
        if 'f1' in metrics:
            f1_score = f1(prediction, ground_truth)
            f1_scores.append(f1_score)
            
        if 'em' in metrics:
            em_score = ems(prediction, ground_truths)
            em_scores.append(em_score)
    
    # Compute average scores
    results = {
        'file': os.path.basename(predictions_file),
        'total_questions': len(df),
        'evaluated_questions': len(df_evaluated)
    }
    
    if 'f1' in metrics:
        results['f1_score'] = np.mean(f1_scores)
    
    if 'em' in metrics:
        results['exact_match'] = np.mean(em_scores)
    
    return results



def evaluate_all_results_one(results_dir, dataset=None, output_file=None):
    """
    Evaluate all result files in the specified directory.
    Args:
        results_dir: Directory containing result files
        dataset: Optional filter for specific dataset results (list or single string)
        output_file: Path to save evaluation results
    Returns:
        DataFrame with evaluation results for all files
    """
    # Initialize empty list to store all results
    all_results = []
    
    # Handle dataset parameter properly
    datasets = []
    if dataset:
        if isinstance(dataset, str):
            datasets = [dataset]
        else:
            datasets = dataset
    else:
        # If no dataset specified, use the results_dir directly
        datasets = [""]
    
    # Process each dataset
    for data in datasets:
        search_path = os.path.join(results_dir, data) if data else results_dir
        result_files = glob.glob(os.path.join(search_path, "*.csv"))
        
        if not result_files:
            logger.error(f"No result files found in {search_path}")
            continue
            
        logger.info(f"Found {len(result_files)} result files to evaluate in {search_path}")
        
        # Evaluate each file
        for file_path in result_files:
            try:
                result = evaluate_predictions(file_path)
                
                # Extract configuration from filename
                file_name = os.path.basename(file_path)
                config_parts = file_name.replace('.csv', '').split('_')
                
                # Add configuration details to results
                for i in range(0, len(config_parts)-1, 2):
                    if i+1 < len(config_parts):
                        key = config_parts[i]
                        value = config_parts[i+1]
                        if key and value:
                            result[key] = value
                
                # Add dataset information if applicable
                if data:
                    result['dataset'] = data
                
                # Add file path for reference
                result['file_path'] = file_path
                
                all_results.append(result)
                logger.info(f"Evaluated {file_path}: F1={result.get('f1_score', 'N/A'):.4f}, EM={result.get('exact_match', 'N/A'):.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {file_path}: {str(e)}")
    
    # Convert all results to DataFrame
    if not all_results:
        logger.error("No results were successfully evaluated")
        return pd.DataFrame()
        
    results_df = pd.DataFrame(all_results)
    
    # Sort by F1 score (descending)
    if 'f1_score' in results_df.columns:
        results_df = results_df.sort_values('f1_score', ascending=False)
    
    # Save results if output file is specified
    if output_file:
        # Reorder and rename columns
        if 'dataset' in results_df.columns and 'memtype' in results_df.columns and 'file' in results_df.columns:
            results_df = results_df.copy()
            # Save 'file' column temporarily and remove it
            file_col = results_df.pop('file')
            
            # Rearrange: [dataset, memtype, <other cols...>, file]
            front_cols = ['dataset', 'memtype']
            remaining_cols = [col for col in results_df.columns if col not in front_cols]
            results_df = results_df[front_cols + remaining_cols]
            
            # Add 'file' back as the last column
            results_df['file'] = file_col

        results_df.to_csv(output_file, index=False)
        logger.info(f"Cleaned evaluation results saved to {output_file}")









    return results_df







def evaluate_all_results(results_dir, dataset=None, output_file=None):
    """
    Evaluate all result files in the specified directory.
    
    Args:
        results_dir: Directory containing result files
        dataset: Optional filter for specific dataset results
        output_file: Path to save evaluation results
    
    Returns:
        DataFrame with evaluation results for all files
    """
    # Find all result files
    for data in dataset:

        search_path = os.path.join(results_dir, data) if data else results_dir
        result_files = glob.glob(os.path.join(search_path, "*.csv"))
    
        if not result_files:
            logger.error(f"No result files found in {search_path}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(result_files)} result files to evaluate")
        
        # Evaluate each file
        all_results = []
        for file_path in tqdm(result_files, desc="Evaluating files"):
            try:
                result = evaluate_predictions(file_path)
                # Extract configuration from filename
                file_name = os.path.basename(file_path)
                config_parts = file_name.replace('.csv', '').split('_')
                
                # Add configuration details to results
                for i in range(0, len(config_parts)-1, 2):
                    if i+1 < len(config_parts):
                        key = config_parts[i]
                        value = config_parts[i+1]
                        if key and value:
                            result[key] = value
                
                all_results.append(result)
                logger.info(f"Evaluated {file_path}: F1={result.get('f1_score', 'N/A'):.4f}, EM={result.get('exact_match', 'N/A'):.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {file_path}: {str(e)}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Sort by F1 score (descending)
        if 'f1_score' in results_df.columns:
            results_df = results_df.sort_values('f1_score', ascending=False)
        
        # Save results if output file is specified
        if output_file:
            results_df.to_csv(os.path.join(search_path, "eval_output.csv"), index=False)
            logger.info(f"Evaluation results saved to {search_path}")
    




    
    #return results_df








# def evaluate_all_results(results_dir, dataset=None, output_file=None):
#     """
#     Evaluate all result files in the specified directory.
    
#     Args:
#         results_dir: Directory containing result files
#         dataset: Optional filter for specific dataset results
#         output_file: Path to save evaluation results
    
#     Returns:
#         DataFrame with evaluation results for all files
#     """
#     # Find all result files

#     search_path = os.path.join(results_dir, dataset) if dataset else results_dir
#     result_files = glob.glob(os.path.join(search_path, "*.csv"))
    
#     if not result_files:
#         logger.error(f"No result files found in {search_path}")
#         return pd.DataFrame()
    
#     logger.info(f"Found {len(result_files)} result files to evaluate")
    
#     # Evaluate each file
#     all_results = []
#     for file_path in tqdm(result_files, desc="Evaluating files"):
#         try:
#             result = evaluate_predictions(file_path)
#             # Extract configuration from filename
#             file_name = os.path.basename(file_path)
#             config_parts = file_name.replace('.csv', '').split('_')
            
#             # Add configuration details to results
#             for i in range(0, len(config_parts)-1, 2):
#                 if i+1 < len(config_parts):
#                     key = config_parts[i]
#                     value = config_parts[i+1]
#                     if key and value:
#                         result[key] = value
            
#             all_results.append(result)
#             logger.info(f"Evaluated {file_path}: F1={result.get('f1_score', 'N/A'):.4f}, EM={result.get('exact_match', 'N/A'):.4f}")
#         except Exception as e:
#             logger.error(f"Error evaluating {file_path}: {str(e)}")
    
#     # Convert to DataFrame
#     results_df = pd.DataFrame(all_results)
    
#     # Sort by F1 score (descending)
#     if 'f1_score' in results_df.columns:
#         results_df = results_df.sort_values('f1_score', ascending=False)
    
#     # Save results if output file is specified
#     if output_file:
#         results_df.to_csv(output_file, index=False)
#         logger.info(f"Evaluation results saved to {output_file}")
    
#     return results_df

def print_best_configurations(results_df, top_n=5):
    """
    Print the best performing configurations based on F1 score.
    
    Args:
        results_df: DataFrame with evaluation results
        top_n: Number of top configurations to print
    """
    if len(results_df) == 0:
        logger.warning("No results to display")
        return
    
    logger.info(f"\n===== Top {min(top_n, len(results_df))} Configurations =====")
    
    display_columns = ['file', 'f1_score', 'exact_match', 'evaluated_questions', 'total_questions']
    config_columns = [col for col in results_df.columns if col not in display_columns]
    
    for i, (_, row) in enumerate(results_df.head(top_n).iterrows()):
        logger.info(f"\nRank {i+1}:")
        logger.info(f"F1 Score: {row.get('f1_score', 'N/A'):.4f}")
        logger.info(f"Exact Match: {row.get('exact_match', 'N/A'):.4f}")
        logger.info(f"Evaluated: {row['evaluated_questions']}/{row['total_questions']} questions")
        
        logger.info("Configuration:")
        for col in config_columns:
            if col in row and pd.notna(row[col]):
                logger.info(f"  - {col}: {row[col]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA model predictions")
    parser.add_argument("--results_dir", type=str, default=[REDACTED], 
                        help="Directory containing result files")
    #'quality'
    # parser.add_argument("--dataset", type=str, default=None,
[REDACTED]
    #                     help="Filter results by dataset")
[REDACTED]
                        help="Filter results by dataset")

    parser.add_argument("--output_file", type=str, default=[REDACTED],
                        help="Path to save evaluation results")
    parser.add_argument("--single_file", type=str, default=None,
                        help="Evaluate a single result file instead of a directory")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Number of top configurations to display")
    
    args = parser.parse_args()
    
    
    
    #-----------------------------
    #args.single_file= [REDACTED]
    
    
    # python eval.py --results_dir [REDACTED] --output_file [REDACTED]
    
    
    if args.single_file:
        # Evaluate a single file
        if not os.path.exists(args.single_file):
            logger.error(f"File not found: {args.single_file}")
            exit(1)
            
        result = evaluate_predictions(args.single_file)
        logger.info(f"Evaluation results for {args.single_file}:")
        logger.info(f"F1 Score: {result.get('f1_score', 'N/A'):.4f}")
        logger.info(f"Exact Match: {result.get('exact_match', 'N/A'):.4f}")
        logger.info(f"Evaluated: {result['evaluated_questions']}/{result['total_questions']} questions")
        
        if args.output_file:
            pd.DataFrame([result]).to_csv(args.output_file, index=False)
            logger.info(f"Results saved to {args.output_file}")
    else:
        # Evaluate all files in the directory
        #results_df = evaluate_all_results(args.results_dir, args.dataset, args.output_file)
        results_df = evaluate_all_results_one(args.results_dir, args.dataset, args.output_file)
        
        #print_best_configurations(results_df, args.top_n)
        
        # Print overall statistics
        # if len(results_df) > 0:
        #     logger.info("\n===== Overall Statistics =====")
        #     logger.info(f"Total configurations evaluated: {len(results_df)}")
        #     logger.info(f"Average F1 Score: {results_df['f1_score'].mean():.4f}")
        #     logger.info(f"Average Exact Match: {results_df['exact_match'].mean():.4f}")
        #     logger.info(f"Best F1 Score: {results_df['f1_score'].max():.4f}")
        #     logger.info(f"Best Exact Match: {results_df['exact_match'].max():.4f}")