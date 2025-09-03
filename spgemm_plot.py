import matplotlib.pyplot as plt
import numpy as np

# Dataset information with nodes, edges, and improvement percentages
# Data from Matrix Data table in your project documentation
dataset_info = {
    'reddit': {'nodes': 232965, 'edges': 114615891, 'improvement': 37.58668},
    'flickr': {'nodes': 89250, 'edges': 989006, 'improvement': 33.06},
    'yelp': {'nodes': 716847, 'edges': 13954819, 'improvement': 25.923},
    'ogbn-proteins': {'nodes': 132534, 'edges': 79122504, 'improvement': 20.522},
    'arxiv': {'nodes': 169343, 'edges': 1166243, 'improvement': 29.59},
    'products': {'nodes': 2449029, 'edges': 123718280, 'improvement': 56.4}
}

# Sort datasets by number of edges (small to large)
sorted_datasets = sorted(dataset_info.items(), key=lambda x: x[1]['edges'])

# Extract sorted data
datasets = [item[0].capitalize() for item in sorted_datasets]
improvements = [item[1]['improvement'] for item in sorted_datasets]
edges = [item[1]['edges'] for item in sorted_datasets]
nodes = [item[1]['nodes'] for item in sorted_datasets]

# Set up the plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Create bars for improvement percentage (left y-axis)
colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71', '#e67e22']
bars = ax1.bar(datasets, improvements, color=colors, alpha=0.8, edgecolor='black', 
               linewidth=1.5, width=0.6, label='AIA Improvement (%)')

# Configure left y-axis (improvement percentage)
ax1.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=10)
ax1.set_xlabel('Dataset (sorted by graph size)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(improvements) * 1.15)

# Add improvement value labels on top of bars
for bar, value in zip(bars, improvements):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, 
             f'{value:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Create second y-axis for graph size (number of edges)
ax2 = ax1.twinx()
ax2.set_ylabel('Number of Edges (log scale)', fontsize=12, fontweight='bold', color='darkred')
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor='darkred', labelsize=10)

# Plot graph size as line with markers
line = ax2.plot(datasets, edges, color='darkred', marker='o', markersize=8, 
                linewidth=2, alpha=0.7, label='Graph Size (edges)')

# Add edge count labels
for i, (dataset, edge_count) in enumerate(zip(datasets, edges)):
    if edge_count >= 1e6:
        label = f'{edge_count/1e6:.1f}M'
    elif edge_count >= 1e3:
        label = f'{edge_count/1e3:.0f}K'
    else:
        label = f'{edge_count:,}'
    
    ax2.text(i, edge_count * 1.3, label, ha='center', va='bottom', 
             fontsize=9, color='darkred', fontweight='bold')

# Customize the plot
ax1.set_title('SpGEMM AIA Improvement vs Graph Size\n(Datasets sorted by number of edges)', 
              fontsize=14, fontweight='bold', pad=20)

# Add grid for better readability
ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

# Customize x-axis
ax1.tick_params(axis='x', labelsize=11, rotation=45)

# Add average line for improvement
avg_improvement = np.mean(improvements)
ax1.axhline(y=avg_improvement, color='red', linestyle=':', alpha=0.7, linewidth=2)
ax1.text(len(datasets) - 0.5, avg_improvement + 1.5, 
         f'Avg: {avg_improvement:.1f}%', ha='right', va='bottom', 
         fontsize=10, color='red', fontweight='bold')

# Add legends
ax1.legend(loc='upper left', fontsize=10)
ax2.legend(loc='upper right', fontsize=10)

# Add summary statistics text box
summary_text = f"""Dataset Statistics:
• Smallest: {datasets[0]} ({edges[0]:,} edges)
• Largest: {datasets[-1]} ({edges[-1]:,} edges)
• Size range: {edges[-1]/edges[0]:.0f}x difference
• Best improvement: {datasets[np.argmax(improvements)]} ({max(improvements):.1f}%)
• Avg improvement: {avg_improvement:.1f}%"""

ax1.text(0.02, 0.72, summary_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# Save the plots
plt.savefig('aia_improvement_spgemm_sorted.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('aia_improvement_spgemm_sorted.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Print the sorted data table
print("\nSpGEMM AIA Improvement Data (Sorted by Graph Size):")
print("="*70)
print(f"{'Dataset':<12} {'Nodes':<10} {'Edges':<12} {'Improvement (%)':<15}")
print("="*70)
for i, dataset in enumerate(datasets):
    print(f"{dataset:<12} {nodes[i]:<10,} {edges[i]:<12,} {improvements[i]:<15.3f}")
print("="*70)
print(f"{'Average':<12} {np.mean(nodes):<10,.0f} {np.mean(edges):<12,.0f} {avg_improvement:<15.2f}")
print(f"{'Std Dev':<12} {np.std(nodes):<10,.0f} {np.std(edges):<12,.0f} {np.std(improvements):<15.2f}")

# Additional analysis
print(f"\nGraph Size Analysis:")
print(f"• Smallest graph: {datasets[0]} with {edges[0]:,} edges")
print(f"• Largest graph: {datasets[-1]} with {edges[-1]:,} edges") 
print(f"• Size ratio (largest/smallest): {edges[-1]/edges[0]:.0f}x")
print(f"• Improvement correlation with size: {np.corrcoef(edges, improvements)[0,1]:.3f}")

print(f"\nImprovement Analysis:")
print(f"• Best improvement: {max(improvements):.3f}% ({datasets[np.argmax(improvements)]})")
print(f"• Lowest improvement: {min(improvements):.3f}% ({datasets[np.argmin(improvements)]})")
print(f"• Improvement range: {max(improvements) - min(improvements):.3f}%")
print(f"• Median improvement: {np.median(improvements):.2f}%")
