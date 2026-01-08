#!/usr/bin/env python3
"""
Demo: Generate Natural Language Coaching Report
================================================
This script demonstrates how to generate natural language coaching feedback
from KSI evaluation results.

Usage:
    python demo_nlp_coaching.py --type hybrid [--skill-level intermediate]
"""

import argparse
import os
import sys
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, 'src')

from ksi_v2 import EnhancedKSI
from natural_language_coach import generate_coaching_report

def demo_coaching_from_data(pipeline_type='hybrid', skill_level='intermediate'):
    """
    Load preprocessed data and generate coaching report for a sample.
    """
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    cfg = params[f'{pipeline_type}_pipeline']
    template_path = params['expert_pipeline']['output_path']
    
    # Check prerequisites
    if not os.path.exists(cfg['data_path']):
        print(f"❌ Data path not found: {cfg['data_path']}")
        return
    
    if not os.path.exists(template_path):
        print(f"❌ Template file not found: {template_path}")
        print("   Run: python src/generate_templates.py")
        return
    
    print("="*70)
    print("NATURAL LANGUAGE COACHING DEMO")
    print("="*70)
    print(f"Pipeline: {pipeline_type}")
    print(f"Skill Level: {skill_level}")
    print("="*70)
    
    # Load data
    classes = sorted(os.listdir(cfg['data_path']))
    print(f"\nAvailable shot types: {classes}")
    
    # Pick first class with data
    for cls in classes:
        cls_path = os.path.join(cfg['data_path'], cls)
        if not os.path.isdir(cls_path):
            continue
        
        files = [f for f in os.listdir(cls_path) if f.endswith('.npz')]
        if not files:
            continue
        
        # Load one sample
        sample_file = os.path.join(cls_path, files[0])
        data = np.load(sample_file)
        
        print(f"\n📊 Analyzing: {cls} - {files[0]}")
        
        # Get landmarks
        if 'raw_landmarks' in data:
            user_landmarks = data['raw_landmarks']
            print(f"   Landmarks shape: {user_landmarks.shape}")
        else:
            print("   ⚠️  No raw landmarks in data. Cannot generate KSI report.")
            print("   Run preprocessing with raw landmark saving enabled.")
            continue
        
        # Load template
        templates = np.load(template_path, allow_pickle=True)
        
        # Find template
        template_key = None
        if cls in templates:
            template_key = cls
        elif f'{cls}_variant1' in templates:
            template_key = f'{cls}_variant1'
        
        if template_key is None:
            print(f"   ⚠️  No template found for {cls}")
            continue
        
        expert_template = templates[template_key]
        if expert_template.ndim == 2 and expert_template.shape[1] == 99:
            expert_landmarks = expert_template.reshape(-1, 33, 3)
        else:
            expert_landmarks = expert_template
        
        print(f"   Template shape: {expert_landmarks.shape}")
        
        # Calculate KSI
        print("\n⚙️  Calculating KSI metrics...")
        ksi_calc = EnhancedKSI(fps=30.0)
        
        ksi_result = ksi_calc.calculate(
            expert_landmarks=expert_landmarks,
            user_landmarks=user_landmarks,
            weights={'pose': 0.4, 'velocity': 0.4, 'acceleration': 0.2}
        )
        
        print(f"   ✅ KSI Total: {ksi_result.ksi_total:.3f}")
        print(f"   ✅ KSI Weighted: {ksi_result.ksi_weighted:.3f}")
        
        # Generate coaching report
        print("\n📝 Generating Natural Language Coaching Report...")
        report = generate_coaching_report(
            ksi_result=ksi_result,
            shot_type_str=cls,
            skill_level_str=skill_level,
            user_name="Demo User",
            output_format='text'
        )
        
        # Save report
        os.makedirs("coaching_reports", exist_ok=True)
        report_path = f"coaching_reports/demo_{cls}_{skill_level}.txt"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Saved report to: {report_path}")
        print("\n" + "="*70)
        print("COACHING REPORT PREVIEW")
        print("="*70)
        
        # Print first 50 lines
        lines = report.split('\n')
        for line in lines[:50]:
            print(line)
        
        if len(lines) > 50:
            print(f"\n... ({len(lines) - 50} more lines) ...")
            print(f"\nRead full report at: {report_path}")
        
        print("\n" + "="*70)
        
        # Generate JSON version
        json_report = generate_coaching_report(
            ksi_result=ksi_result,
            shot_type_str=cls,
            skill_level_str=skill_level,
            output_format='json'
        )
        
        json_path = f"coaching_reports/demo_{cls}_{skill_level}.json"
        with open(json_path, 'w') as f:
            f.write(json_report)
        
        print(f"📄 JSON version: {json_path}")
        print("="*70)
        
        # Only process one sample for demo
        break
    else:
        print("\n❌ No valid samples found with raw landmarks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo natural language coaching generation")
    parser.add_argument("--type", choices=['pose', 'hybrid'], default='hybrid',
                        help="Pipeline type (default: hybrid)")
    parser.add_argument("--skill-level", type=str, default='intermediate',
                        choices=['beginner', 'intermediate', 'advanced', 'expert'],
                        help="User skill level (default: intermediate)")
    
    args = parser.parse_args()
    
    demo_coaching_from_data(args.type, args.skill_level)
