import pandas as pd
import json
from rag_engine import RAGEngine
from typing import Dict, List

def load_train_data(path="data/train.csv") -> Dict[str, List[str]]:
   
    df = pd.read_csv(path)
    
    # Group URLs by query
    grouped = df.groupby('Query')['Assessment_url'].apply(list).to_dict()
    
    print(f"📚 Loaded {len(grouped)} training queries")
    return grouped

def calculate_recall_at_k(predicted_urls: List[str], 
                          ground_truth_urls: List[str], 
                          k: int = 10) -> float:
    """
    Calculate Recall@K metric
    
    Recall@K = (Number of relevant items in top-K) / (Total relevant items)
    
    Args:
        predicted_urls: List of URLs returned by system (in rank order)
        ground_truth_urls: List of correct URLs for this query
        k: Number of top results to consider
    
    Returns:
        Recall score between 0 and 1
    """
    predicted_set = set(predicted_urls[:k])
    relevant_set = set(ground_truth_urls)
    
    if len(relevant_set) == 0:
        return 0.0
    
    # Count how many relevant items we retrieved
    hits = len(predicted_set.intersection(relevant_set))
    
    return hits / len(relevant_set)

def evaluate_engine(engine: RAGEngine, train_data: Dict[str, List[str]]) -> float:
    """
    Evaluate RAG engine on training data
    
    Returns: Mean Recall@10 score
    """
    print("\n" + "="*60)
    print("🧪 EVALUATION ON TRAIN SET")
    print("="*60 + "\n")
    
    recalls = []
    
    for query, ground_truth_urls in train_data.items():
        # Get predictions
        results = engine.process_query(query)
        predicted_urls = [r['url'] for r in results]
        
        # Calculate recall
        recall = calculate_recall_at_k(predicted_urls, ground_truth_urls, k=10)
        recalls.append(recall)
        
        # Detailed output
        print(f"Query: {query[:60]}...")
        print(f"  Ground Truth: {len(ground_truth_urls)} assessments")
        print(f"  Predicted: {len(predicted_urls)} assessments")
        print(f"  Recall@10: {recall:.3f}")
        print(f"  Hits: {len(set(predicted_urls).intersection(set(ground_truth_urls)))}")
        print()
    
    # Compute mean
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    print("="*60)
    print(f"📊 FINAL SCORE: Mean Recall@10 = {mean_recall:.4f}")
    print("="*60 + "\n")
    
    return mean_recall

def generate_test_predictions(engine: RAGEngine, 
                              test_queries_path: str = "data/test.csv",
                              output_path: str = "predictions.csv"):
    """
    Generate predictions for unlabeled test set
    
    Creates CSV in required format:
    Query,Assessment_url
    Query 1,URL 1
    Query 1,URL 2
    ...
    """
    print("\n" + "="*60)
    print("🎯 GENERATING TEST SET PREDICTIONS")
    print("="*60 + "\n")
    
    # Load test queries
    test_df = pd.read_csv(test_queries_path)
    
    rows = []
    for idx, query in enumerate(test_df['Query'], 1):
        print(f"[{idx}/{len(test_df)}] Processing: {query[:50]}...")
        
        # Get recommendations
        results = engine.process_query(query)
        
        # Add to output (top 10)
        for result in results[:10]:
            rows.append({
                'Query': query,
                'Assessment_url': result['url']
            })
    
    # Save CSV
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Predictions saved to {output_path}")
    print(f"   Total rows: {len(output_df)}")
    print(f"   Format: Query, Assessment_url")

def main():
    """
    Main evaluation workflow:
    1. Initialize RAG engine
    2. Evaluate on train set
    3. Generate test predictions
    """
    print("🚀 Starting Evaluation Pipeline\n")
    
    # Initialize engine (this will trigger scraping if needed)
    engine = RAGEngine()
    
    # Evaluate on train set
    try:
        train_data = load_train_data("data/train.csv")
        mean_recall = evaluate_engine(engine, train_data)
        
        if mean_recall < 0.3:
            print("⚠ WARNING: Low recall score. Consider:")
            print("   - Improving query balancing prompt")
            print("   - Adjusting result balancing logic")
            print("   - Using different embedding model")
    
    except FileNotFoundError:
        print("⚠ train.csv not found. Skipping training evaluation.")
        print("   Download from assignment link and place in backend/data/")
    
    # Generate test predictions
    try:
        generate_test_predictions(engine, "data/test.csv", "predictions.csv")
    
    except FileNotFoundError:
        print("⚠ test.csv not found. Skipping test predictions.")
        print("   Download from assignment link and place in backend/data/")

if __name__ == "__main__":
    main()
