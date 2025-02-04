def feature_statistics_template(feature_name,
                                mean,
                                std,
                                min_v,
                                max_v,
                                feature_units,
                                template_manager):
    ""
    # Check if feature has a unit
    unit = ""
    if feature_name.lower() in feature_units.keys():
        unit = feature_units[feature_name.lower()]

    if len(unit) > 0:
        mean = mean + " " + unit
        std = std + " " + unit
        min_v = min_v + " " + unit
        max_v = max_v + " " + unit

    """return (f"Here are statistics for the feature <b>{feature_name}</b>: <br><br>"
            f"The <b>mean</b> is {mean}<br><br> one <b>standard deviation</b> is {std}<br><br>"
            f" the <b>minimum</b> value is {min_v}<br><br> and the <b>maximum</b> value is {max_v}.")"""
    feature_name = template_manager.get_feature_display_name_by_name(feature_name)
    response_text = f"<b>{feature_name}</b> ranges from {min_v} to {max_v} with a mean of {mean}."

    return response_text
