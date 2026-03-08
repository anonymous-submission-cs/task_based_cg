import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


def plot_tsne_with_equivalence_class_score(
    df,
    interested_test_keys,
    equivalent_train_keys,
    remaining_train_keys,
    target_equivalence_class_map,
    target_equivalence_class_label,
    test_combination_accs,
    label,
):
    width = 1200
    height = 800
    point_size = 8
    alpha_val = 0.8

    def mask_for(highlight_set):
        return df["original_perm"].isin(highlight_set)

    def get_equivalent_score_for_train(train_key, target_equivalence_class_map):
        # find the test key in the equivalence_class_map which has the train_key
        for test_key, value in target_equivalence_class_map.items():
            if train_key in value.keys() and value[train_key] > 0:
                return value[train_key]
        return 0

    fig = go.Figure()
    mask1 = mask_for(interested_test_keys)
    mask2 = mask_for(equivalent_train_keys)
    mask3 = mask_for(remaining_train_keys)
    

    # Plot Level 3: light gray diamonds (background)
    if mask3.any():
        mask3_subset = df[mask3]

        fig.add_trace(
            go.Scattergl(
                x=mask3_subset["x"],
                y=mask3_subset["y"],
                mode="markers",
                name="Remaining Train Tasks",
                marker=dict(
                    symbol="diamond", color="lightgray", size=point_size, opacity=0.25
                ),
                hovertext=[" ".join(list(x)) for x in mask3_subset["original_perm"]],
                showlegend=True,
            )
        )

    # Plot Level 2: blue squares with color legend
    if mask2.any():
        mask2_subset = df[mask2]
        mask2_subset["equivalence_class_score"] = mask2_subset["original_perm"].apply(
            lambda x: get_equivalent_score_for_train(
                tuple(str(k) for k in x), target_equivalence_class_map
            )
        )
        # normalize the equivalence_class_score to 0-1
        hovertext = [" ".join(list(x)) for x in mask2_subset["original_perm"]]
        hovertext = [
            f"{h} (Equivalence Score: {s:.2f}, x: {x}, y_actual: {y_actual}, y_pred: {y_pred})"
            for h, s, x, y_actual, y_pred in zip(
                hovertext,
                mask2_subset["equivalence_class_score"],
                mask2_subset["input_string"],
                mask2_subset["actual_output_string"],
                mask2_subset["predicted_output_string"],
                
            )
        ]
        fig.add_trace(
            go.Scattergl(
                x=mask2_subset["x"],
                y=mask2_subset["y"],
                mode="markers",
                name="Equivalent Train Tasks",
                marker=dict(
                    symbol="square",
                    color=mask2_subset["equivalence_class_score"],
                    size=point_size,
                    opacity=alpha_val,
                    line=dict(width=0.6, color="black"),
                    colorscale="Blues",
                    colorbar=dict(
                        title="Equivalence Class Score",
                        x=1.15,
                        y=0.65,
                        len=0.4,
                        thickness=15
                    ),
                    showscale=True
                ),
                hovertext=hovertext,
                showlegend=True,
            )
        )

    # Plot Level 1: red circles with color legend
    if mask1.any():
        mask1_subset = df[mask1]
        mask1_subset["test_score"] = mask1_subset["original_perm"].apply(
            lambda x: test_combination_accs.get(tuple(str(k) for k in x), 0)
        )
        hovertext = [" ".join(list(x)) for x in mask1_subset["original_perm"]]
        hovertext = [
            f"{h} (Acc: {s:.2f}, x: {x}, y_actual: {y_actual}, y_pred: {y_pred})"
            for h, s, x, y_actual, y_pred in zip(
                hovertext,
                mask1_subset["test_score"],
                mask1_subset["input_string"],
                mask1_subset["actual_output_string"],
                mask1_subset["predicted_output_string"],
            )
        ]
        fig.add_trace(
            go.Scattergl(
                x=mask1_subset["x"],
                y=mask1_subset["y"],
                mode="markers",
                name="Test Task",
                marker=dict(
                    symbol="circle",
                    color=mask1_subset["test_score"], 
                    colorscale=[[0, "white"], [1, "red"]],
                    size=point_size,
                    opacity=alpha_val,
                    line=dict(width=0.8, color="black"),
                    colorbar=dict(
                        title="Test Accuracy",
                        x=1.15,
                        y=0.15,
                        len=0.4,
                        thickness=15
                    ),
                    showscale=True
                ),
                hovertext=hovertext,
                showlegend=True,
            )
        )

    fig.update_layout(
        title=f"Representation of Test Tasks in t-SNE based on {target_equivalence_class_label}",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        width=width,
        height=height,
        legend=dict(
            title="Legend",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(r=300, l=50, t=50, b=50),  # Increased right margin for color bars
    )
    
    print(f"tsne_plot_{label}.html")
    fig.write_html(f"tsne_plot_{label}.html")


def plot_tsne_with_equivalence_class_score_with_slider(
    df,
    interested_test_keys,
    equivalent_train_keys,
    remaining_train_keys,
    target_equivalence_class_map,
    target_equivalence_class_label,
    test_combination_accs,
    label,
):
    width = 1400  # Increased width for checkboxes
    height = 800
    point_size = 8
    alpha_val = 0.8

    def mask_for(highlight_set):
        return df["original_perm"].isin(highlight_set)

    def get_equivalent_score_for_train(train_key, target_equivalence_class_map):
        # find the test key in the equivalence_class_map which has the train_key
        for test_key, value in target_equivalence_class_map.items():
            if train_key in value.keys():
                return value[train_key]
        return 0

    def get_equivalent_tasks_for_test(test_key, target_equivalence_class_map):
        """Get all equivalent train tasks for a given test key"""
        if test_key in target_equivalence_class_map:
            # filter out the keys which have equivalence_class_score > 0
            return list(filter(lambda x: target_equivalence_class_map[test_key][x] > 0, target_equivalence_class_map[test_key].keys()))
        return []

    def get_test_key_string(test_key):
        """Convert test key to string representation"""
        if isinstance(test_key, (list, tuple)):
            return " ".join(str(k) for k in test_key)
        return str(test_key)

    fig = go.Figure()
    mask1 = mask_for(interested_test_keys)
    mask2 = mask_for(equivalent_train_keys)
    mask3 = mask_for(remaining_train_keys)
    
    # Store test key accuracies for checkbox labels
    test_key_accuracies = {}
    if mask1.any():
        mask1_subset = df[mask1].copy()
        
        for _, row in mask1_subset.iterrows():
            test_key = tuple(str(k) for k in row["original_perm"])
            test_key_accuracies[test_key] = test_combination_accs.get(test_key, 0)
        
        # Compute correct_flag for sample-level accuracy
        mask1_subset["correct_flag"] = mask1_subset["actual_output_string"] == mask1_subset["predicted_output_string"]

    # Get list of interested test keys for checkbox creation
    interested_test_keys_list = list(interested_test_keys)
    num_test_keys = len(interested_test_keys_list)

    # Plot Level 3: light gray diamonds (background) - multiple traces for test key filtering
    if mask3.any():
        mask3_subset = df[mask3]
        
        # Create separate traces for each test key filter
        for j in range(num_test_keys + 1):  # +1 for "all test keys"
            fig.add_trace(
                go.Scattergl(
                    x=mask3_subset["x"],
                    y=mask3_subset["y"],
                    mode="markers",
                    name="Remaining Train Tasks",
                    marker=dict(
                        symbol="diamond", color="lightgray", size=point_size, opacity=0.25
                    ),
                    hovertext=[" ".join(list(x)) for x in mask3_subset["original_perm"]],
                    showlegend=True if j == 0 else False,
                    visible=True if j == 0 else False,
                )
            )

    # Plot Level 2: blue squares - multiple traces for test key filtering
    if mask2.any():
        mask2_subset = df[mask2].copy()
        mask2_subset["equivalence_class_score"] = mask2_subset["original_perm"].apply(
            lambda x: get_equivalent_score_for_train(
                tuple(str(k) for k in x), target_equivalence_class_map
            )
        )
        # Create traces for different test key filters
        for j in range(num_test_keys + 1):  # +1 for "all test keys"
            # Find which equivalent tasks should be highlighted
            if j == 0:  # Show all test keys
                relevant_test_keys = list(target_equivalence_class_map.keys())
            else:  # Show specific test key
                specific_test_key = interested_test_keys_list[j-1]
                test_key_tuple = tuple(str(k) for k in specific_test_key)
                relevant_test_keys = [test_key_tuple]
            
            # Get equivalent tasks for the relevant test keys
            highlighted_equivalent_tasks = []
            for test_key in relevant_test_keys:
                highlighted_equivalent_tasks.extend(get_equivalent_tasks_for_test(test_key, target_equivalence_class_map))
            
            # Filter mask2_subset to only show highlighted equivalent tasks
            if highlighted_equivalent_tasks:
                filtered_mask2 = mask2_subset[mask2_subset["original_perm"].apply(
                    lambda x: tuple(str(k) for k in x) in highlighted_equivalent_tasks
                )]
            else:
                filtered_mask2 = mask2_subset if j == 0 else mask2_subset.iloc[:0]  # Show all for first case, empty for others
            
            hovertext = [" ".join(list(x)) for x in filtered_mask2["original_perm"]]
            hovertext = [
                f"{h} (Equivalence Score: {s:.2f}, x: {x}, y_actual: {y_actual}, y_pred: {y_pred})"
                for h, s, x, y_actual, y_pred in zip(
                    hovertext,
                    filtered_mask2["equivalence_class_score"],
                    filtered_mask2["input_string"],
                    filtered_mask2["actual_output_string"],
                    filtered_mask2["predicted_output_string"]
                    
                )
            ]
            
            fig.add_trace(
                go.Scattergl(
                    x=filtered_mask2["x"],
                    y=filtered_mask2["y"],
                    mode="markers",
                    name="Equivalent Train Tasks",
                    marker=dict(
                        symbol="square",
                        color=filtered_mask2["equivalence_class_score"],
                        size=point_size,
                        opacity=alpha_val,
                        line=dict(width=0.6, color="black"),
                        colorscale="Blues",
                        colorbar=dict(
                            title="Equivalence Class Score",
                            x=1.15,
                            y=0.65,
                            len=0.4,
                            thickness=15
                        ) if j == 0 else None,
                        showscale=True if j == 0 else False
                    ),
                    hovertext=hovertext,
                    showlegend=True if j == 0 else False,
                    visible=True if j == 0 else False,
                )
            )

    # Plot Level 1a: Test points colored by TASK ACCURACY (default)
    if mask1.any():
        mask1_subset = df[mask1].copy()
        mask1_subset["test_score"] = mask1_subset["original_perm"].apply(
            lambda x: test_combination_accs.get(tuple(str(k) for k in x), 0)
        )
        mask1_subset["correct_flag"] = mask1_subset["actual_output_string"] == mask1_subset["predicted_output_string"]
        
        for j in range(num_test_keys + 1):  # +1 for "all test keys"
            # Filter test tasks
            if j == 0:  # Show all test keys
                filtered_mask1 = mask1_subset
            else:  # Show specific test key
                specific_test_key = interested_test_keys_list[j-1]
                test_key_mask = mask1_subset["original_perm"].apply(lambda x: list(x) == list(specific_test_key))
                filtered_mask1 = mask1_subset[test_key_mask]
            
            hovertext = [" ".join(list(x)) for x in filtered_mask1["original_perm"]]
            hovertext = [
                f"{h} (Task Acc: {s:.2f}, Sample Correct: {c}, x: {x}, y_actual: {y_actual}, y_pred: {y_pred})"
                for h, s, c, x, y_actual, y_pred in zip(
                    hovertext,
                    filtered_mask1["test_score"],
                    filtered_mask1["correct_flag"],
                    filtered_mask1["input_string"],
                    filtered_mask1["actual_output_string"],
                    filtered_mask1["predicted_output_string"],
                )
            ]
            
            fig.add_trace(
                go.Scattergl(
                    x=filtered_mask1["x"],
                    y=filtered_mask1["y"],
                    mode="markers",
                    name="Test Task (Task Accuracy)",
                    marker=dict(
                        symbol="circle",
                        color=filtered_mask1["test_score"], 
                        colorscale=[[0, "white"], [1, "red"]],
                        size=point_size,
                        opacity=alpha_val,
                        line=dict(width=0.8, color="black"),
                        colorbar=dict(
                            title="Test Accuracy",
                            x=1.15,
                            y=0.15,
                            len=0.4,
                            thickness=15
                        ) if j == 0 else None,
                        showscale=True if j == 0 else False
                    ),
                    hovertext=hovertext,
                    showlegend=True if j == 0 else False,
                    visible=True if j == 0 else False,
                )
            )

    # Plot Level 1b: Test points colored by SAMPLE CORRECTNESS (binary)
    if mask1.any():
        for j in range(num_test_keys + 1):  # +1 for "all test keys"
            # Filter test tasks  
            if j == 0:  # Show all test keys
                filtered_mask1 = mask1_subset
            else:  # Show specific test key
                specific_test_key = interested_test_keys_list[j-1]
                test_key_mask = mask1_subset["original_perm"].apply(lambda x: list(x) == list(specific_test_key))
                filtered_mask1 = mask1_subset[test_key_mask]
            
            hovertext = [" ".join(list(x)) for x in filtered_mask1["original_perm"]]
            hovertext = [
                f"{h} (Task Acc: {s:.2f}, Sample Correct: {c}, x: {x}, y_actual: {y_actual}, y_pred: {y_pred})"
                for h, s, c, x, y_actual, y_pred in zip(
                    hovertext,
                    filtered_mask1["test_score"],
                    filtered_mask1["correct_flag"],
                    filtered_mask1["input_string"],
                    filtered_mask1["actual_output_string"],
                    filtered_mask1["predicted_output_string"],
                )
            ]
            
            fig.add_trace(
                go.Scattergl(
                    x=filtered_mask1["x"],
                    y=filtered_mask1["y"],
                    mode="markers",
                    name="Test Task (Sample Correctness)",
                    marker=dict(
                        symbol="circle",
                        color=filtered_mask1["correct_flag"].astype(int), 
                        colorscale=[[0, "white"], [1, "red"]],
                        size=point_size,
                        opacity=alpha_val,
                        line=dict(width=0.8, color="black"),
                        colorbar=dict(
                            title="Sample Correct (0=Wrong, 1=Right)",
                            x=1.15,
                            y=0.15,
                            len=0.4,
                            thickness=15
                        ) if j == 0 else None,
                        showscale=True if j == 0 else False,
                    ),
                    hovertext=hovertext,
                    showlegend=True if j == 0 else False,
                    visible=True if j == 0 else False,  # Hidden by default
                )
            )

    # Create test key checkboxes
    test_key_buttons = []
    if interested_test_keys_list:
        total_traces = len(fig.data)
        traces_per_test_key = 4  # remaining, equivalent, test_task_acc, test_sample_acc
        
        # Add "All Test Keys" button
        all_visible = [False] * total_traces
        for level in range(4):  # All 4 levels
            trace_idx = level * (num_test_keys + 1) + 0  # j=0 for "all"
            if trace_idx < total_traces:
                all_visible[trace_idx] = True
        
        test_key_buttons.append(dict(
            label="All Test Keys",
            method="restyle",
            args=[{"visible": all_visible}]
        ))
        
        # Add individual test key buttons
        for idx, test_key in enumerate(interested_test_keys_list):
            test_key_str = get_test_key_string(test_key)
            test_key_tuple = tuple(str(k) for k in test_key)
            accuracy = test_key_accuracies.get(test_key_tuple, 0.0)
            
            # Create visibility array for this specific test key
            key_visible = [False] * total_traces
            for level in range(4):  # All 4 levels
                trace_idx = level * (num_test_keys + 1) + (idx + 1)  # j=idx+1 for specific key
                if trace_idx < total_traces:
                    key_visible[trace_idx] = True
            
            test_key_buttons.append(dict(
                label=f"{test_key_str} (Acc: {accuracy:.2f})",
                method="restyle",
                args=[{"visible": key_visible}]
            ))

    # Task accuracy vs sample correctness toggle
    total_traces = len(fig.data)
    color_toggle_buttons = [
        dict(
            label="Task Accuracy",
            method="restyle",
            args=[{
                "visible": [True if i < total_traces // 4 * 3 else False for i in range(total_traces)]
            }]
        ),
        dict(
            label="Sample Correctness",
            method="restyle", 
            args=[{
                "visible": [True if i >= total_traces // 4 * 3 or i < total_traces // 4 * 2 else False for i in range(total_traces)]
            }]
        )
    ]

    fig.update_layout(
        title=f"Representation of Test Tasks in t-SNE based on {target_equivalence_class_label}",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        width=width,
        height=height,
        legend=dict(
            title="Legend",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(r=400, l=50, t=120, b=120),  # Increased right margin for checkboxes
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                buttons=test_key_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.25,
                xanchor="left",
                y=0.9,
                yanchor="top",
                name="Test Key & Color Mode Filter"
            )
        ] if test_key_buttons else []
    )
    
    print(f"tsne_plot_{label}.html")
    fig.write_html(f"tsne_plot_{label}.html")