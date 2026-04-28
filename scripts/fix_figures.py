"""
Adds plt.close(fig) before every 'return fig' that follows a savefig+print pair
in sft_master.py, to prevent matplotlib's 'too many open figures' warning.
"""
import re

path = r'c:/Users/Shivansh Mishra/Downloads/files (24)/sft_master.py'
content = open(path, encoding='utf-8').read()

# Pattern: savefig → print([saved]) → return fig
pattern = (
    r"(    plt\.savefig\(save_path, dpi=_STYLE\['fig_dpi'\], bbox_inches='tight'\)\n"
    r"    print\(f\"  \[saved\] \{save_path\}\"\)\n)"
    r"    return fig"
)
replacement = r"\1    plt.close(fig)\n    return fig"

new_content, n = re.subn(pattern, replacement, content)
print(f'Replacements made: {n}')
open(path, 'w', encoding='utf-8').write(new_content)
print('Done')
