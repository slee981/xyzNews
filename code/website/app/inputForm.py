from wtforms import Form, TextAreaField, validators, SubmitField


class InputForm(Form):

    # Starting seed
    article = TextAreaField(
        label="Paste an article here:",
        validators=[validators.InputRequired("Please paste an article")],
    )

    # Submit button
    submit = SubmitField("Enter")
