#!/usr/bin/env python3
"""
Update LaTeX paper with actual experimental results.
Reads JSON result files and replaces placeholder values in main.tex.
"""
import os
import json
import re

CKPT_DIR = 'checkpoints'
LATEX_FILE = 'latex/main.tex'


def load_results():
    """Load all result files."""
    results = {}
    for name in ['test_results', 'perturbation_results', 'adversarial_results',
                  'cross_domain_results', 'dice_results']:
        path = os.path.join(CKPT_DIR, f'{name}.json')
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
            print(f'  Loaded {name}.json')
        else:
            print(f'  Missing {name}.json')
    return results


def update_table(tex_content, results):
    """Replace placeholder table with actual values."""
    if 'test_results' not in results:
        return tex_content

    tr = results['test_results']

    # Build replacement table
    r4 = tr.get('4', {})
    r8 = tr.get('8', {})

    new_table = f"""\\begin{{table}}[t]
\\caption{{Reconstruction quality on the MM-WHS cardiac MR test set.}}\\label{{tab:recon}}
\\centering
\\begin{{tabular}}{{l|ccc|ccc}}
\\toprule
& \\multicolumn{{3}}{{c|}}{{R=4$\\times$}} & \\multicolumn{{3}}{{c}}{{R=8$\\times$}} \\\\
Method & PSNR$\\uparrow$ & SSIM$\\uparrow$ & NMSE$\\downarrow$ & PSNR$\\uparrow$ & SSIM$\\uparrow$ & NMSE$\\downarrow$ \\\\
\\midrule
Zero-filled & {r4.get('zf_psnr', 0):.2f} & {r4.get('zf_ssim', 0):.4f} & -- & {r8.get('zf_psnr', 0):.2f} & {r8.get('zf_ssim', 0):.4f} & -- \\\\
Ours (U-Net+DC) & \\textbf{{{r4.get('psnr', 0):.2f}}} & \\textbf{{{r4.get('ssim', 0):.4f}}} & \\textbf{{{r4.get('nmse', 0):.6f}}} & \\textbf{{{r8.get('psnr', 0):.2f}}} & \\textbf{{{r8.get('ssim', 0):.4f}}} & \\textbf{{{r8.get('nmse', 0):.6f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    # Replace old table
    old_pattern = r'\\begin\{table\}\[t\]\s*\\caption\{Reconstruction quality.*?\\end\{table\}'
    tex_content = re.sub(old_pattern, new_table, tex_content, flags=re.DOTALL)

    # Remove note about placeholder
    tex_content = tex_content.replace(
        '\\textit{Note: Exact values will be populated from experimental results.}\n', '')

    return tex_content


def update_implementation_details(tex_content, results):
    """Update implementation details section with actual Optuna results."""
    optuna_file = os.path.join(CKPT_DIR, 'optuna_best_params_R4.json')
    if os.path.exists(optuna_file):
        with open(optuna_file) as f:
            optuna = json.load(f)
        bp = optuna['best_params']
        # The current text already has the correct Optuna values from initial write
    return tex_content


def main():
    print('Updating LaTeX with experimental results...')
    results = load_results()

    with open(LATEX_FILE) as f:
        tex = f.read()

    tex = update_table(tex, results)
    tex = update_implementation_details(tex, results)

    with open(LATEX_FILE, 'w') as f:
        f.write(tex)

    print(f'Updated {LATEX_FILE}')

    # Also print summary
    if 'test_results' in results:
        print('\n--- Test Results Summary ---')
        for R, data in sorted(results['test_results'].items(), key=lambda x: int(x[0])):
            print(f"  R={R}x: PSNR={data['psnr']:.2f}±{data['psnr_std']:.2f}, "
                  f"SSIM={data['ssim']:.4f}±{data['ssim_std']:.4f}")

    if 'cross_domain_results' in results:
        cd = results['cross_domain_results']
        print(f"\n--- Cross-Domain ---")
        print(f"  MR: PSNR={cd['mr_psnr']:.2f}, Unc={cd['mr_unc']:.6f}")
        print(f"  CT: PSNR={cd['ct_psnr']:.2f}, Unc={cd['ct_unc']:.6f}")

    if 'dice_results' in results:
        print(f"\n--- Downstream Segmentation ---")
        for R, data in sorted(results['dice_results'].items(), key=lambda x: int(x[0])):
            print(f"  R={R}x: GT={data['gt_dice']:.4f}, Recon={data['recon_dice']:.4f}, ZF={data['zf_dice']:.4f}")


if __name__ == '__main__':
    main()
