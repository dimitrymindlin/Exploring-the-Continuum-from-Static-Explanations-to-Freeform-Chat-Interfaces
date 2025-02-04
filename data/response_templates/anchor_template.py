def anchor_template(exp, template_manager):
    output_string = ""
    # output_string += "By fixing all of the following attributes, the prediction stays the same even though other attributes are changed:"
    output_string += "<br><br>"

    display_names_exp_names = []
    for idx, change_string in enumerate(exp.names()):
        feature = change_string.split(" ")[0].strip()
        try:
            value = change_string.split("=")[1].strip()
        except IndexError:
            # Check if < or > is in the string and split
            if "<" in change_string:
                value = change_string.split("<")[1].strip()
            elif "<=" in change_string:
                value = change_string.split("<=")[1].strip()
            elif ">" in change_string:
                value = change_string.split(">")[1].strip()
            elif ">=" in change_string:
                value = change_string.split(">=")[1].strip()
            else:
                raise ValueError("Could not split the change string.")
        # If its a categorical feature, get the display name of the value
        """if feature in template_manager.encoded_col_mapping.keys(): NOT NEEDED?!
            value = template_manager.get_encoded_feature_name(feature, value)"""
        display_name = template_manager.get_feature_display_name_by_name(feature)
        precision = exp.precision()
        precision_string = f"{precision * 100:.2f}%"
        coverage = exp.coverage()
        coverage_string = f"{coverage * 100:.2f}%"
        #exp_string = f"For {display_name} {change_string.split(' ')[1]} {value}, the prediction is {precision_string} accurate and this condition applies for {coverage_string} of similar cases."
        exp_string = f"{display_name} {change_string.split(' ')[1]} {value}"
        display_names_exp_names.append(exp_string)

    # Initialize an empty list to hold the updated display names
    updated_display_names = []

    # Turn mathematical symbols into words and update the list
    for explanation in display_names_exp_names:
        explanation = explanation.replace("<=", "is not above")
        explanation = explanation.replace(">=", "is not below")
        explanation = explanation.replace(">", "is above")
        explanation = explanation.replace("<", "is below")
        explanation = explanation.replace(".00", "")  # remove decimal zeroes
        # Add the updated explanation to the new list
        updated_display_names.append(explanation)

    # Wrap each updated name/expression in <b> and </b> for bold
    bold_display_names = ["<b>{}</b>".format(name) for name in updated_display_names]

    # Join with " and ", keeping "and" not bold
    explanation_text = " and ".join(bold_display_names)
    return explanation_text
