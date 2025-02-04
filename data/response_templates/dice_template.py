def textual_cf(cfe_strings):
    output_string = ""

    for i, cfe_string in enumerate(cfe_strings, 1):
        output_string += f"{i}. {cfe_string}<br>"

    return output_string
