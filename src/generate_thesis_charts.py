import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for academic publication
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.autolayout': True
})

def generate_context_limit_chart():
    """Generates latency comparison across varying top_k context limits."""
    limits = [1, 3, 5]
    latencies = [385, 542, 715] # Measured empirical averages in ms
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=limits, y=latencies, palette="Blues_d")
    plt.title("Retrieval Latency vs. Context Window Size (top_k)")
    plt.xlabel("Context Limit (top_k)")
    plt.ylabel("Total Latency (ms)")
    
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ms", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3), 
                    textcoords='offset points')
        
    plt.savefig("context_limit_latency.png", dpi=300)
    plt.close()
    print("[+] Generated: context_limit_latency.png")

def generate_model_comparison_chart():
    """Generates performance score comparison with and without Rerankers."""
    metrics = ['Faithfulness', 'Answer Relevance', 'Context Precision']
    mini_lm_no_rerank = [0.78, 0.81, 0.74]
    mini_lm_with_rerank = [0.89, 0.92, 0.88]
    mpnet_with_rerank = [0.93, 0.95, 0.91]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, mini_lm_no_rerank, width, label='MiniLM (No Rerank)', color='#a6bddb')
    plt.bar(x, mini_lm_with_rerank, width, label='MiniLM + Reranker', color='#3182bd')
    plt.bar(x + width, mpnet_with_rerank, width, label='MPNet + Reranker', color='#08519c')
    
    plt.title("RAG Evaluation Metrics Across Model Configurations")
    plt.xticks(x, metrics)
    plt.ylabel("Evaluation Score (0 - 1)")
    plt.ylim(0.5, 1.0)
    plt.legend(loc='lower right')
    
    plt.savefig("model_comparison_performance.png", dpi=300)
    plt.close()
    print("[+] Generated: model_comparison_performance.png")

def generate_cache_threshold_chart():
    """Generates semantic cache hit rate curves across thresholds."""
    thresholds = [0.85, 0.90, 0.92, 0.95, 0.98]
    hit_rates = [0.92, 0.85, 0.78, 0.62, 0.35]
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, hit_rates, marker='o', linewidth=2.5, markersize=8, color='#2ca02c')
    plt.title("Semantic Cache Hit Rate vs. Similarity Threshold")
    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("Hit Rate Ratio")
    plt.ylim(0.0, 1.05)
    
    for i, txt in enumerate(hit_rates):
        plt.annotate(f"{txt:.2f}", (thresholds[i], hit_rates[i] + 0.03), ha='center', fontsize=11)
        
    plt.savefig("cache_threshold_hitrate.png", dpi=300)
    plt.close()
    print("[+] Generated: cache_threshold_hitrate.png")

if __name__ == "__main__":
    print("Generating Master's Thesis Evaluation Charts...")
    generate_context_limit_chart()
    generate_model_comparison_chart()
    generate_cache_threshold_chart()
    print("All charts successfully generated and saved to root directory!")
