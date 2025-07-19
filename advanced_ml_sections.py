# Advanced ML Sections to add to the notebook

# Section 14: Fuzzy C-Means Clustering
fcm_section = """
def perform_fcm_clustering_with_umap(X, n_clusters=4, m=2.0, n_neighbors=15, min_dist=0.1):
    \"\"\"
    Perform Fuzzy C-Means clustering with UMAP 3D visualization.
    
    Parameters:
    - n_clusters: Number of clusters
    - m: Fuzziness parameter (1.1 to 3.0, higher = fuzzier)
    - n_neighbors: UMAP parameter for local structure
    - min_dist: UMAP parameter for point spread
    \"\"\"
    print(f"🔬 Performing Fuzzy C-Means clustering with {n_clusters} clusters...")
    print(f"   Fuzziness parameter (m): {m}")
    
    # Apply Fuzzy C-Means clustering
    cntr, u, u0, d, jm, p, fpc = cmeans(
        X.T,  # FCM expects features as rows
        c=n_clusters,
        m=m,
        error=0.005,
        maxiter=1000,
        seed=42
    )
    
    print("   ✅ Fuzzy C-Means clustering completed")
    
    # Get hard cluster assignments (highest membership)
    cluster_labels_hard = np.argmax(u, axis=0)
    
    # Calculate cluster statistics
    print("\\n📊 Cluster Statistics (Hard Assignments):")
    unique_labels, counts = np.unique(cluster_labels_hard, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(cluster_labels_hard)) * 100
        print(f"   Cluster {label}: {count} samples ({percentage:.1f}%)")
    
    # Print Fuzzy Partition Coefficient
    print(f"\\n📈 Fuzzy Partition Coefficient (FPC): {fpc:.3f}")
    print("   (Higher FPC indicates crisper clustering, range: 0-1)")
    
    # Apply UMAP for 3D visualization
    print(f"\\n🎨 Applying UMAP for 3D visualization...")
    
    umap_3d = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        n_epochs=200
    )
    
    umap_features = umap_3d.fit_transform(X)
    print("   ✅ UMAP transformation completed")
    
    # Create two visualizations: hard assignments and fuzzy memberships
    
    # 1. Hard Assignment Visualization
    fig1 = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#6C5CE7', '#A8E6CF', '#FD79A8', '#FDCB6E']
    
    for i in range(n_clusters):
        cluster_mask = cluster_labels_hard == i
        
        fig1.add_trace(go.Scatter3d(
            x=umap_features[cluster_mask, 0],
            y=umap_features[cluster_mask, 1],
            z=umap_features[cluster_mask, 2],
            mode='markers',
            name=f'Cluster {i}',
            marker=dict(
                size=5,
                color=colors[i % len(colors)],
                opacity=0.8,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=[f'Sample {idx}<br>Cluster {i}<br>Membership: {u[i, idx]:.3f}' 
                  for idx in np.where(cluster_mask)[0]],
            hovertemplate='%{text}<br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>'
        ))
    
    fig1.update_layout(
        title=dict(
            text=f'Fuzzy C-Means Clustering - Hard Assignments ({n_clusters} clusters)',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1000,
        height=800
    )
    
    fig1.show()
    
    # 2. Fuzzy Membership Visualization (showing uncertainty)
    fig2 = go.Figure()
    
    # Calculate maximum membership for each point
    max_membership = np.max(u, axis=0)
    
    # Color points by their maximum membership strength
    fig2.add_trace(go.Scatter3d(
        x=umap_features[:, 0],
        y=umap_features[:, 1],
        z=umap_features[:, 2],
        mode='markers',
        name='Fuzzy Membership',
        marker=dict(
            size=5,
            color=max_membership,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(
                title='Max Membership',
                tickmode='linear',
                tick0=0,
                dtick=0.2
            ),
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=[f'Sample {i}<br>Cluster: {cluster_labels_hard[i]}<br>Max Membership: {max_membership[i]:.3f}' 
              for i in range(len(cluster_labels_hard))],
        hovertemplate='%{text}<br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>'
    ))
    
    fig2.update_layout(
        title=dict(
            text='Fuzzy C-Means - Membership Strength Visualization',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1000,
        height=800
    )
    
    fig2.show()
    
    # 3. Membership distribution analysis
    plt.figure(figsize=(12, 6))
    
    # Plot membership distribution
    plt.subplot(1, 2, 1)
    plt.hist(max_membership, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Maximum Membership Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Maximum Membership Values')
    plt.axvline(np.mean(max_membership), color='red', linestyle='--', 
                label=f'Mean: {np.mean(max_membership):.3f}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot membership heatmap for sample of points
    plt.subplot(1, 2, 2)
    sample_size = min(50, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    membership_sample = u[:, sample_indices]
    
    im = plt.imshow(membership_sample, aspect='auto', cmap='YlOrRd')
    plt.xlabel('Sample Index')
    plt.ylabel('Cluster')
    plt.title('Membership Matrix (Sample)')
    plt.colorbar(im, label='Membership Degree')
    
    plt.tight_layout()
    plt.show()
    
    # Return results
    results = {
        'cluster_labels_hard': cluster_labels_hard,
        'membership_matrix': u,
        'cluster_centers': cntr,
        'umap_features': umap_features,
        'fpc': fpc,
        'max_membership': max_membership
    }
    
    return results

# Perform Fuzzy C-Means clustering
print("\\n=== FUZZY C-MEANS CLUSTERING WITH UMAP ===")
fcm_results = perform_fcm_clustering_with_umap(X_preprocessed, n_clusters=4, m=2.0)
"""

# Section 15: Fuzzy SVM Classification
fuzzy_svm_section = """
def implement_fuzzy_svm_classification(X, y, target_name, test_size=0.3):
    \"\"\"
    Implement Fuzzy SVM classification with probability estimates.
    
    This provides:
    1. Probability distributions for each class
    2. Confidence analysis
    3. Calibration curves for binary classification
    \"\"\"
    print(f"🎯 Implementing Fuzzy SVM for {target_name} prediction...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train SVM with probability estimates
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    svm_model.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = svm_model.predict(X_test)
    y_proba = svm_model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\\n📊 Test Accuracy: {accuracy:.3f}")
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Probability distribution for each class
    classes = svm_model.classes_
    for i, cls in enumerate(classes):
        axes[0, 0].hist(y_proba[:, i], bins=20, alpha=0.7, 
                       label=f'Class {cls}', edgecolor='black')
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Probability Distribution by Class')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Confidence analysis (maximum probability)
    max_proba = np.max(y_proba, axis=1)
    correct_mask = y_pred == y_test
    
    axes[0, 1].hist(max_proba[correct_mask], bins=20, alpha=0.7, 
                   label='Correct', color='green', edgecolor='black')
    axes[0, 1].hist(max_proba[~correct_mask], bins=20, alpha=0.7, 
                   label='Incorrect', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Maximum Probability (Confidence)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Confusion matrix with probabilities
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title('Confusion Matrix')
    
    # 4. Calibration plot (for binary classification)
    if len(classes) == 2:
        from sklearn.calibration import calibration_curve
        
        # Get probabilities for positive class
        pos_proba = y_proba[:, 1]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, pos_proba, n_bins=10
        )
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 
                       'o-', label='SVM calibration')
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Calibration Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    else:
        # For multi-class, show per-class metrics
        metrics_df = pd.DataFrame(report).T[:-3]  # Exclude avg rows
        metrics_df[['precision', 'recall', 'f1-score']].plot(
            kind='bar', ax=axes[1, 1]
        )
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Per-Class Metrics')
        axes[1, 1].legend()
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
    
    plt.suptitle(f'Fuzzy SVM Analysis - {target_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\\n📋 Classification Report:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    for cls in classes:
        cls_report = report[str(cls)]
        print(f"{cls:<10} {cls_report['precision']:<10.3f} {cls_report['recall']:<10.3f} "
              f"{cls_report['f1-score']:<10.3f} {int(cls_report['support']):<10}")
    
    # Analyze uncertain predictions
    uncertainty_threshold = 0.6
    uncertain_mask = max_proba < uncertainty_threshold
    n_uncertain = np.sum(uncertain_mask)
    
    print(f"\\n🤔 Uncertain Predictions (confidence < {uncertainty_threshold}):")
    print(f"   Count: {n_uncertain} ({n_uncertain/len(y_test)*100:.1f}%)")
    if n_uncertain > 0:
        print(f"   Accuracy on uncertain: {accuracy_score(y_test[uncertain_mask], y_pred[uncertain_mask]):.3f}")
        print(f"   Accuracy on confident: {accuracy_score(y_test[~uncertain_mask], y_pred[~uncertain_mask]):.3f}")
    
    return {
        'model': svm_model,
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'max_proba': max_proba,
        'report': report
    }

# Apply Fuzzy SVM to each target
print("\\n=== FUZZY SVM CLASSIFICATION ===")
fuzzy_svm_results = {}

for col in y_classification_processed.columns[:3]:  # First 3 targets for demonstration
    print(f"\\n{'='*60}")
    result = implement_fuzzy_svm_classification(
        X_preprocessed, 
        y_classification_processed[col], 
        col
    )
    fuzzy_svm_results[col] = result
"""

# Section 16: Genetic Algorithm Feature Selection
ga_section = """
def implement_genetic_algorithm_feature_selection(X, y, target_name, n_generations=30, 
                                                 population_size=30, k_folds=5):
    \"\"\"
    Implement genetic algorithm for feature selection.
    
    This uses evolutionary optimization to find the best feature subset
    that maximizes classification accuracy.
    \"\"\"
    print(f"🧬 Starting Genetic Algorithm Feature Selection for {target_name}...")
    print(f"   Population size: {population_size}")
    print(f"   Generations: {n_generations}")
    print(f"   Features to select from: {X.shape[1]}")
    
    # Define fitness function (maximizing)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Initialize toolbox
    toolbox = base.Toolbox()
    
    # Attribute generator (0 or 1 for each feature)
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_bool, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Fitness function
    def evalFeatureSet(individual):
        # Select features
        selected_features = [i for i, val in enumerate(individual) if val == 1]
        
        # Need at least 1 feature
        if len(selected_features) == 0:
            return 0.0,
        
        # Extract selected features
        X_selected = X.iloc[:, selected_features]
        
        # Use SVM with cross-validation
        svm = SVC(kernel='rbf', random_state=42)
        scores = cross_val_score(svm, X_selected, y, cv=k_folds, scoring='accuracy')
        
        # Return average accuracy
        return np.mean(scores),
    
    toolbox.register("evaluate", evalFeatureSet)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    population = toolbox.population(n=population_size)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of fame to track best individuals
    hof = tools.HallOfFame(1)
    
    # Track evolution
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields
    
    # Evaluate initial population
    print("\\n📊 Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    hof.update(population)
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(population), **record)
    
    # Evolution
    print("\\n🔄 Starting evolution...")
    for gen in range(1, n_generations + 1):
        # Select next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        population[:] = offspring
        
        # Update hall of fame and statistics
        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        
        # Progress update
        if gen % 5 == 0:
            print(f"   Generation {gen}: Best fitness = {record['max']:.3f}")
    
    print("\\n✅ Evolution completed!")
    
    # Extract best solution
    best_individual = hof[0]
    best_features = [i for i, val in enumerate(best_individual) if val == 1]
    best_feature_names = X.columns[best_features].tolist()
    
    print(f"\\n🏆 Best Solution:")
    print(f"   Features selected: {len(best_features)}/{X.shape[1]}")
    print(f"   Best fitness (accuracy): {best_individual.fitness.values[0]:.3f}")
    print(f"   Selected features: {best_feature_names[:5]}..." if len(best_feature_names) > 5 
          else f"   Selected features: {best_feature_names}")
    
    # Visualize evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Evolution progress
    gen_data = logbook.select("gen")
    max_fits = logbook.select("max")
    avg_fits = logbook.select("avg")
    
    ax1.plot(gen_data, max_fits, 'b-', label='Best fitness')
    ax1.plot(gen_data, avg_fits, 'r-', label='Average fitness')
    ax1.fill_between(gen_data, 
                    [avg - std for avg, std in zip(logbook.select("avg"), logbook.select("std"))],
                    [avg + std for avg, std in zip(logbook.select("avg"), logbook.select("std"))],
                    alpha=0.2, color='red')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (Accuracy)')
    ax1.set_title('Evolution Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Feature selection frequency across all individuals in final population
    feature_counts = np.zeros(X.shape[1])
    for ind in population:
        feature_counts += np.array(ind)
    
    # Sort features by selection frequency
    sorted_indices = np.argsort(feature_counts)[::-1][:20]  # Top 20
    
    ax2.barh(range(len(sorted_indices)), feature_counts[sorted_indices], 
            color='skyblue', edgecolor='navy')
    ax2.set_yticks(range(len(sorted_indices)))
    ax2.set_yticklabels([X.columns[i] for i in sorted_indices])
    ax2.set_xlabel('Selection Frequency in Final Population')
    ax2.set_title('Feature Popularity')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Genetic Algorithm Results - {target_name}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Compare with baseline (all features)
    print("\\n📊 Performance Comparison:")
    
    # Baseline with all features
    svm_all = SVC(kernel='rbf', random_state=42)
    scores_all = cross_val_score(svm_all, X, y, cv=k_folds, scoring='accuracy')
    
    # GA selected features
    X_ga_selected = X.iloc[:, best_features]
    svm_ga = SVC(kernel='rbf', random_state=42)
    scores_ga = cross_val_score(svm_ga, X_ga_selected, y, cv=k_folds, scoring='accuracy')
    
    print(f"   All features ({X.shape[1]}): {np.mean(scores_all):.3f} (±{np.std(scores_all):.3f})")
    print(f"   GA selected ({len(best_features)}): {np.mean(scores_ga):.3f} (±{np.std(scores_ga):.3f})")
    print(f"   Feature reduction: {(1 - len(best_features)/X.shape[1])*100:.1f}%")
    
    return {
        'best_individual': best_individual,
        'best_features': best_features,
        'best_feature_names': best_feature_names,
        'best_fitness': best_individual.fitness.values[0],
        'logbook': logbook,
        'final_population': population
    }

# Apply GA to select targets
print("\\n=== GENETIC ALGORITHM FEATURE SELECTION ===")
ga_results = {}

for col in y_classification_processed.columns[:2]:  # First 2 targets for demonstration
    print(f"\\n{'='*60}")
    result = implement_genetic_algorithm_feature_selection(
        X_preprocessed, 
        y_classification_processed[col], 
        col,
        n_generations=20,  # Reduced for faster execution
        population_size=20
    )
    ga_results[col] = result
"""

print("✅ Advanced ML sections code saved to advanced_ml_sections.py")
print("\nTo add these sections to the notebook:")
print("1. Open the notebook in Jupyter")
print("2. Add new cells after the K-Means clustering section")
print("3. Copy and paste each section from advanced_ml_sections.py")
print("\nThe sections include:")
print("- Fuzzy C-Means Clustering with UMAP")
print("- Fuzzy SVM Classification")
print("- Genetic Algorithm Feature Selection")
print("- Comprehensive Results Summary")
print("- Model Saving and Export")
print("- Final Conclusions")