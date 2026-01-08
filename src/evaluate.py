# Updated evaluate.py
import argparse
import os
import yaml
import numpy as np
import json
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from typing import Dict
# Local Imports - use ksi_v2 with enhanced features
from ksi_v2 import (
    EnhancedKSI,
    ShotPhase,
)
from mlflow_utils import MLflowRunManager  # <--- NEW: Enhanced MLflow utilities
try:
    from natural_language_coach import generate_coaching_report
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("⚠️  natural_language_coach not available. NLP feedback will be skipped.")


def evaluate(pipeline_type: str, model_path: str = None, generate_nlp_feedback: bool = False, 
             nlp_skill_level: str = 'intermediate', max_nlp_samples: int = 5, auto_run_name: bool = False,
             data_path: str = None):
    """
    Evaluate model with enhanced KSI v2 metrics.
    Logs phase scores, confidence intervals, ranking hinge, and component breakdowns.
    
    Args:
        pipeline_type: 'pose' or 'hybrid'
        model_path: Optional path to model file (overrides params.yaml)
        generate_nlp_feedback: Whether to generate natural language coaching reports
        nlp_skill_level: Skill level for NLP coach ('beginner', 'intermediate', 'advanced', 'expert')
        max_nlp_samples: Maximum number of samples to generate detailed feedback for
        auto_run_name: If True, auto-generate MLflow run name without prompting
        data_path: Optional path to evaluation data (overrides params.yaml)
    """
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    cfg = params[f'{pipeline_type}_pipeline']
    ksi_cfg = params.get('ksi', {'weights': {'pose': 0.5, 'velocity': 0.3, 'acceleration': 0.2}})
    
    # Use provided model path or default from config
    if model_path:
        cfg['model_path'] = model_path
        run_suffix = f" (custom: {os.path.basename(model_path)})"
    else:
        run_suffix = ""
    
    # Use provided data path or default from config
    if data_path:
        cfg['data_path'] = data_path
    
    # Setup MLflow with interactive run manager
    exp_name = "Pose_LSTM_Experiment" if pipeline_type == 'pose' else "Hybrid_TCN_Experiment"
    run_manager = MLflowRunManager(exp_name)

    with run_manager.start_interactive_run(
        default_description=f"Evaluation of {pipeline_type} pipeline with KSI v2.0 metrics{run_suffix}",
        auto_name=auto_run_name
    ):
        # Log evaluation configuration
        mlflow.log_param("evaluation.model_path", cfg['model_path'])
        mlflow.log_param("evaluation.data_path", cfg['data_path'])
        mlflow.log_param("evaluation.pipeline_type", pipeline_type)
        
        # Display what we're evaluating on
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION CONFIGURATION")
        print(f"{'='*70}")
        print(f"Pipeline: {pipeline_type}")
        print(f"Model: {cfg['model_path']}")
        print(f"Data: {cfg['data_path']}")
        print(f"{'='*70}\n")
        
        # 1. Load Data
        X, y = [], []
        raw_landmarks = []  # NEW: store raw landmarks for KSI
        if not os.path.exists(cfg['data_path']):
            print(f"Data path {cfg['data_path']} not found")
            return
        classes = sorted(os.listdir(cfg['data_path']))
        for i, cls in enumerate(classes):
            path = os.path.join(cfg['data_path'], cls)
            if not os.path.isdir(path):
                continue
            for f in os.listdir(path):
                if f.endswith('.npz'):
                    data = np.load(os.path.join(path, f))
                    X.append(data['features'])
                    y.append(i)
                    # NEW: load raw landmarks if available (for hybrid pipeline KSI)
                    if 'raw_landmarks' in data:
                        raw_landmarks.append(data['raw_landmarks'])
                    else:
                        raw_landmarks.append(None)
        
        if not X:
            print("No data loaded")
            return
        X, y_cat = np.array(X), to_categorical(y, len(classes))
        
        # Split data and raw landmarks together
        if raw_landmarks and any(lm is not None for lm in raw_landmarks):
            _, X_test, _, y_test, _, raw_test = train_test_split(
                X, y_cat, raw_landmarks, test_size=0.2, stratify=y, random_state=42
            )
            has_raw_landmarks = True
        else:
            _, X_test, _, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)
            raw_test = None
            has_raw_landmarks = False
        
        # 2. Load Model & Predict
        model = load_model(cfg['model_path'])
        
        if pipeline_type == 'hybrid':
            cnn_dim = cfg['cnn_feature_dim']
            X_pose, X_cnn = X_test[..., :-cnn_dim], X_test[..., -cnn_dim:]
            loss, acc = model.evaluate([X_cnn, X_pose], y_test, verbose=0)
            y_pred = np.argmax(model.predict([X_cnn, X_pose]), axis=1)
            sample_pose = X_pose
        else:
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            y_pred = np.argmax(model.predict(X_test), axis=1)
            sample_pose = X_test

        # 3. Enhanced KSI v2 Calculation with contact-centered windowing
        # NEW: KSI now works with hybrid pipeline if raw_landmarks are available
        template_path = params['expert_pipeline']['output_path']
        ksi_results = {
            'avg_ksi_total': 0.0,
            'avg_ksi_weighted': 0.0,
            'avg_confidence_ci_width': 0.0,
            'avg_uncertainty_scalar': 0.0,
            'phase_scores': {},
            'component_scores': {},
            'reliable_count': 0,
            'total_samples': 0
        }
        
        # Check if we can do KSI evaluation
        can_do_ksi = os.path.exists(template_path) and (
            pipeline_type == 'pose' or (pipeline_type == 'hybrid' and has_raw_landmarks)
        )
        
        if can_do_ksi:
            print(f"Expert templates found at {template_path}. Computing KSI metrics...")
            templates = np.load(template_path, allow_pickle=True)
            
            # Check template format - detect if raw landmarks or enhanced features
            sample_template_key = next((k for k in templates.files if not k.startswith('_')), None)
            if sample_template_key is None:
                print(f"⚠ No valid templates found in {template_path}")
                can_do_ksi = False
            else:
                sample_template = templates[sample_template_key]
                # Raw landmarks: (T, 33, 3) or (T, 99) flattened
                # Enhanced features: (T, 32)
                template_is_raw_landmarks = (
                    sample_template.ndim == 3 and sample_template.shape[1:] == (33, 3)
                ) or (
                    sample_template.ndim == 2 and sample_template.shape[1] == 99
                )
                template_is_enhanced = sample_template.ndim == 2 and sample_template.shape[1] == 32
                
                if template_is_enhanced:
                    print(f"⚠ Templates are in 32-feature format (KSI v2 features), not raw landmarks.")
                    print(f"   KSI calculation requires raw landmarks (T, 33, 3). Regenerate templates with raw landmarks.")
                    can_do_ksi = False
                elif not template_is_raw_landmarks:
                    print(f"⚠ Unknown template format: shape={sample_template.shape}. Expected (T, 33, 3) or (T, 99).")
                    can_do_ksi = False
                else:
                    print(f"   Template format: {'(T, 33, 3)' if sample_template.ndim == 3 else '(T, 99)'} - OK")
        
        if can_do_ksi:
            # Initialize enhanced KSI calculator
            ksi_calc = EnhancedKSI(
                fps=params.get('fps', 30.0),
                contact_window_pre_frames=18,
                contact_window_post_frames=18,
                bootstrap_min=50,
                bootstrap_max=200,
                ranking_margin=0.05
            )
            
            ksi_totals = []
            ksi_weighted_list = []
            ci_widths = []
            uncertainty_scalars = []
            phase_score_accumulator = {p.value: [] for p in ShotPhase}
            component_accumulator = {'pose': [], 'velocity': [], 'acceleration': [], 'jerk': []}
            reliable_count = 0
            
            # Store individual results for NLP feedback
            individual_results = [] if generate_nlp_feedback else None
            
            # Sample for speed (evaluate up to 50 items)
            n_samples = min(50, len(sample_pose))
            evaluated_count = 0  # Track actually evaluated samples
            skipped_templates = set()  # Track missing templates to report once
            
            for i in range(n_samples):
                cls = classes[np.argmax(y_test[i])]
                
                # NEW: Try main template, then variants, prioritizing main
                template_key = None
                if cls in templates:
                    template_key = cls
                elif f'{cls}_variant1' in templates:
                    # If only variants exist, use variant1 (best quality)
                    template_key = f'{cls}_variant1'
                else:
                    # Try any key containing the class name
                    for key in templates.files:
                        if cls in key and not key.startswith('_'):
                            template_key = key
                            break
                
                if template_key is None:
                    if cls not in skipped_templates:
                        skipped_templates.add(cls)
                    continue
                
                # Get user landmarks - use raw_landmarks if available (hybrid), else reshape pose features
                if pipeline_type == 'hybrid' and raw_test is not None and raw_test[i] is not None:
                    user_lm = raw_test[i]  # Already (T, 33, 3)
                else:
                    # Pose pipeline: reshape from flattened features (T, 99) -> (T, 33, 3)
                    try:
                        user_lm = sample_pose[i].reshape(-1, 33, 3)
                    except ValueError as e:
                        print(f"⚠ Cannot reshape pose features to landmarks: {sample_pose[i].shape} -> (T, 33, 3)")
                        continue
                
                # Load expert template and reshape if needed
                expert_template = templates[template_key]
                try:
                    if expert_template.ndim == 3 and expert_template.shape[1:] == (33, 3):
                        expert_lm = expert_template  # Already (T, 33, 3)
                    elif expert_template.ndim == 2 and expert_template.shape[1] == 99:
                        expert_lm = expert_template.reshape(-1, 33, 3)  # (T, 99) -> (T, 33, 3)
                    else:
                        print(f"⚠ Cannot convert template '{template_key}' shape {expert_template.shape} to landmarks")
                        continue
                except ValueError as e:
                    print(f"⚠ Template reshape failed for '{template_key}': {e}")
                    continue
                
                # Calculate enhanced KSI
                result = ksi_calc.calculate(
                    expert_landmarks=expert_lm,
                    user_landmarks=user_lm,
                    weights=ksi_cfg['weights'],
                    baseline_ksi=None  # Could pass previous user score for ranking hinge
                )
                
                ksi_totals.append(result.ksi_total)
                ksi_weighted_list.append(result.ksi_weighted)
                
                # Confidence metrics
                if result.confidence:
                    ci_width = result.confidence.get('ci_95_upper', 0) - result.confidence.get('ci_95_lower', 0)
                    ci_widths.append(ci_width)
                    uncertainty_scalars.append(result.confidence.get('uncertainty_scalar', 0))
                    if result.confidence.get('reliable', False):
                        reliable_count += 1
                
                # Phase scores
                for phase, score in result.phase_scores.items():
                    if phase in phase_score_accumulator:
                        phase_score_accumulator[phase].append(score)
                
                # Component scores
                for comp in ['pose', 'velocity', 'acceleration', 'jerk']:
                    if comp in result.components:
                        component_accumulator[comp].append(result.components[comp])
                
                # Store individual result for NLP feedback (only if prediction is correct)
                if generate_nlp_feedback:
                    predicted_cls = classes[y_pred[i]] if i < len(y_pred) else None
                    if predicted_cls == cls:  # Only store correctly predicted samples
                        individual_results.append({
                            'sample_idx': i,
                            'class': cls,
                            'predicted_class': predicted_cls,
                            'ksi_result': result,
                            'ksi_total': result.ksi_total
                        })
                
                evaluated_count += 1
            
            # Report skipped templates
            if skipped_templates:
                print(f"   ⚠ Missing templates for classes: {sorted(skipped_templates)}")
            
            # Aggregate results
            ksi_results['avg_ksi_total'] = float(np.mean(ksi_totals)) if ksi_totals else 0.0
            ksi_results['avg_ksi_weighted'] = float(np.mean(ksi_weighted_list)) if ksi_weighted_list else 0.0
            ksi_results['avg_confidence_ci_width'] = float(np.mean(ci_widths)) if ci_widths else 0.0
            ksi_results['avg_uncertainty_scalar'] = float(np.mean(uncertainty_scalars)) if uncertainty_scalars else 0.0
            ksi_results['reliable_count'] = reliable_count
            ksi_results['total_samples'] = evaluated_count  # Use actual evaluated count, not attempted
            
            for phase, scores in phase_score_accumulator.items():
                if scores:
                    ksi_results['phase_scores'][phase] = float(np.mean(scores))
            
            for comp, scores in component_accumulator.items():
                if scores:
                    ksi_results['component_scores'][comp] = float(np.mean(scores))
            
            # Generate natural language coaching reports
            if generate_nlp_feedback and individual_results and NLP_AVAILABLE:
                print(f"\n{'='*70}")
                print(f"GENERATING NATURAL LANGUAGE COACHING FEEDBACK")
                print(f"{'='*70}")
                print(f"   Found {len(individual_results)} correctly predicted samples")
                
                # DEBUG: Show KSI scores distribution
                ksi_scores = [r['ksi_total'] for r in individual_results]
                print(f"   KSI scores range: {min(ksi_scores):.3f} - {max(ksi_scores):.3f}")
                
                os.makedirs("coaching_reports", exist_ok=True)
                
                # Pick best sample (highest KSI score) - this gives most interesting feedback
                individual_results.sort(key=lambda x: x['ksi_total'], reverse=True)
                best_sample = individual_results[0]
                samples_to_generate = [best_sample]
                
                sample = samples_to_generate[0]
                cls = sample['class']
                ksi_result = sample['ksi_result']
                sample_idx = sample['sample_idx']
                
                print(f"\n📝 Generating coaching report for: {cls} (KSI: {ksi_result.ksi_total:.3f})")
                
                try:
                    
                    # Generate simplified coaching report
                    report = generate_coaching_report(
                        ksi_result=ksi_result,
                        shot_type_str=cls,
                        skill_level_str=nlp_skill_level,
                        user_name=None,
                        output_format='text',
                        simplified=True  # Remove weekly plan, shorten output
                    )
                    
                    # Save report
                    report_filename = f"coaching_reports/{cls}_ksi{ksi_result.ksi_total:.3f}_report.txt"
                    with open(report_filename, 'w') as f:
                        f.write(report)
                    
                    print(f"   ✅ Saved: {report_filename}")
                    
                    # Also save JSON version
                    json_report = generate_coaching_report(
                        ksi_result=ksi_result,
                        shot_type_str=cls,
                        skill_level_str=nlp_skill_level,
                        output_format='json',
                        simplified=True
                    )
                    json_filename = f"coaching_reports/{cls}_ksi{ksi_result.ksi_total:.3f}_report.json"
                    with open(json_filename, 'w') as f:
                        f.write(json_report)
                    
                    # Log to MLflow
                    mlflow.log_artifact(report_filename)
                    print(f"   📊 Logged to MLflow")
                        
                except Exception as e:
                    print(f"   ⚠️ Failed to generate report: {e}")
                    import traceback
                    traceback.print_exc()
                
                print(f"\n✅ Generated coaching report in coaching_reports/")
                print(f"{'='*70}\n")
        elif pipeline_type == 'hybrid' and not has_raw_landmarks:
            print(f"⚠️  Raw landmarks not found in hybrid data. Run preprocessing again to enable KSI evaluation.")
            print(f"    (Old data format detected - missing 'raw_landmarks' in .npz files)")
        else:
            print(f"⚠ Expert templates not found at {template_path}. Skipping KSI metrics.")

        # 4. Logging to MLflow
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("ksi_total", ksi_results['avg_ksi_total'])
        mlflow.log_metric("ksi_weighted", ksi_results['avg_ksi_weighted'])
        mlflow.log_metric("ksi_ci_width", ksi_results['avg_confidence_ci_width'])
        mlflow.log_metric("ksi_uncertainty", ksi_results['avg_uncertainty_scalar'])
        mlflow.log_metric("ksi_reliable_ratio", 
                          ksi_results['reliable_count'] / max(1, ksi_results['total_samples']))
        
        # Log phase scores
        for phase, score in ksi_results['phase_scores'].items():
            mlflow.log_metric(f"phase_{phase}", score)
        
        # Log component scores
        for comp, score in ksi_results['component_scores'].items():
            mlflow.log_metric(f"ksi_{comp}", score)
        
        # 5. Confusion Matrix
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title(f"Confusion Matrix - {pipeline_type.upper()}")
        
        os.makedirs("dvclive", exist_ok=True)
        # Use production filename for tuned models, pipeline-specific otherwise
        cm_filename = "production_confusion_matrix.png" if "tuned" in cfg['model_path'] else f"{pipeline_type}_confusion_matrix.png"
        cm_path = os.path.join("dvclive", cm_filename)
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # 6. KSI Analysis Plot
        if ksi_results['phase_scores'] or ksi_results['component_scores']:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Phase scores
            if ksi_results['phase_scores']:
                phases = list(ksi_results['phase_scores'].keys())
                p_scores = list(ksi_results['phase_scores'].values())
                axes[0].barh(phases, p_scores, color=['gray', 'blue', 'orange', 'red', 'green'][:len(phases)])
                axes[0].set_xlim([0, 1])
                axes[0].set_xlabel('Average Score')
                axes[0].set_title('Phase Scores (Avg)')
            
            # Component scores
            if ksi_results['component_scores']:
                comps = list(ksi_results['component_scores'].keys())
                c_scores = list(ksi_results['component_scores'].values())
                axes[1].bar(comps, c_scores, color=['steelblue', 'coral', 'seagreen', 'orchid'][:len(comps)])
                axes[1].set_ylim([0, 1])
                axes[1].set_ylabel('Average Score')
                axes[1].set_title('Component Scores (Avg)')
            
            plt.suptitle(f"KSI v2 Analysis - {pipeline_type.upper()} | "
                         f"KSI: {ksi_results['avg_ksi_total']:.3f} ± {ksi_results['avg_confidence_ci_width']:.3f}")
            plt.tight_layout()
            ksi_plot_path = f"dvclive/{pipeline_type}_ksi_analysis.png"
            plt.savefig(ksi_plot_path)
            plt.close()
            mlflow.log_artifact(ksi_plot_path)

        # 7. Save Metrics for DVC
        metrics = {
            "accuracy": float(acc),
            "loss": float(loss),
            "ksi_total": ksi_results['avg_ksi_total'],
            "ksi_weighted": ksi_results['avg_ksi_weighted'],
            "ksi_ci_width": ksi_results['avg_confidence_ci_width'],
            "ksi_uncertainty": ksi_results['avg_uncertainty_scalar'],
            "ksi_reliable_ratio": ksi_results['reliable_count'] / max(1, ksi_results['total_samples']),
            "phase_scores": ksi_results['phase_scores'],
            "component_scores": ksi_results['component_scores']
        }
        
        # Determine metrics filename based on model path
        metrics_filename = "production_metrics.json" if "tuned" in cfg['model_path'] else f"{pipeline_type}_metrics.json"
        metrics_path = os.path.join("dvclive", metrics_filename)
        
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n{'='*60}")
        print(f"{pipeline_type.upper()} EVALUATION (KSI v2)")
        print(f"{'='*60}")
        print(f"Accuracy: {acc:.4f} | Loss: {loss:.4f}")
        print(f"KSI Total: {ksi_results['avg_ksi_total']:.4f}")
        print(f"KSI Weighted: {ksi_results['avg_ksi_weighted']:.4f}")
        print(f"CI Width: {ksi_results['avg_confidence_ci_width']:.4f}")
        print(f"Uncertainty: {ksi_results['avg_uncertainty_scalar']:.4f}")
        print(f"Reliable: {ksi_results['reliable_count']}/{ksi_results['total_samples']}")
        print(f"Phase scores: {ksi_results['phase_scores']}")
        print(f"Component scores: {ksi_results['component_scores']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['pose', 'hybrid'], required=True)
    parser.add_argument("--model", type=str, help="Optional: custom model path (overrides params.yaml)")
    parser.add_argument("--data", type=str, help="Optional: custom data path for evaluation (overrides params.yaml)")
    parser.add_argument("--nlp", action='store_true', help="Generate natural language coaching reports")
    parser.add_argument("--nlp-skill", type=str, default='intermediate', 
                        choices=['beginner', 'intermediate', 'advanced', 'expert'],
                        help="Skill level for natural language feedback (default: intermediate)")
    parser.add_argument("--nlp-samples", type=int, default=5,
                        help="Max number of samples to generate detailed feedback for (default: 5)")
    parser.add_argument("--auto-name", action='store_true', 
                        help="Auto-generate MLflow run name without prompting")
    args = parser.parse_args()
    evaluate(args.type, args.model, args.nlp, args.nlp_skill, args.nlp_samples, args.auto_name, args.data)