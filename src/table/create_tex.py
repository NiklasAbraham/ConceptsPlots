import csv
import os


def create_tex_table(
    csv_path, output_path, title="Table", caption="", section_title="Results"
):
    """
    Create a LaTeX table from a CSV file.

    Parameters:
    csv_path: str
        Path to the input CSV file
    output_path: str
        Path to the output LaTeX file
    title: str
        Title for the table (used in caption)
    caption: str
        Caption text for the table (if empty, uses title)
    section_title: str
        Section title for the document
    """
    # Read CSV file
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty")

    # First row is header
    header = rows[0]
    data_rows = rows[1:]

    num_columns = len(header)

    # Generate column specification (all centered by default)
    col_spec = "c" * num_columns

    # Start building LaTeX document
    latex_content = []
    latex_content.append("\\documentclass[a4paper,10pt]{article}")
    latex_content.append("\\usepackage[margin=1in]{geometry}")
    latex_content.append("\\usepackage{booktabs}")
    latex_content.append("\\usepackage{multirow}")
    latex_content.append("\\usepackage{graphicx}")
    latex_content.append("")
    latex_content.append("\\begin{document}")
    latex_content.append("")
    latex_content.append(f"\\section*{{{section_title}}}")
    latex_content.append("")
    latex_content.append("\\begin{table}[h!]")
    latex_content.append("    \\centering")
    latex_content.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("        \\toprule")

    # Add header row
    header_tex = " & ".join(header)
    latex_content.append(f"        {header_tex} \\\\")
    latex_content.append("        \\midrule")

    # Add data rows
    for row in data_rows:
        # Pad row if necessary
        while len(row) < num_columns:
            row.append("")
        # Escape special LaTeX characters
        row_escaped = [escape_latex(cell) for cell in row[:num_columns]]
        row_tex = " & ".join(row_escaped)
        latex_content.append(f"        {row_tex} \\\\")

    latex_content.append("        \\bottomrule")
    latex_content.append("    \\end{tabular}")

    # Add caption
    if caption:
        latex_content.append(f"    \\caption{{{caption}}}")
    else:
        latex_content.append(f"    \\caption{{{title}.}}")

    latex_content.append("\\end{table}")
    latex_content.append("")
    latex_content.append("\\end{document}")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_content))


def escape_latex(text):
    """
    Escape special LaTeX characters in text.

    Parameters:
    text: str
        Text to escape

    Returns:
    str: Escaped text
    """
    if not isinstance(text, str):
        text = str(text)

    # LaTeX special characters that need escaping
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "^": "\\textasciicircum{}",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text


if __name__ == "__main__":
    # Default paths
    csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
    output_path = os.path.join(os.path.dirname(__file__), "table.tex")

    # You can customize these parameters
    create_tex_table(
        csv_path=csv_path,
        output_path=output_path,
        title="Table",
        caption="",
        section_title="Results",
    )

    print(f"LaTeX table generated: {output_path}")
