Confidence threshold update
============================

Replace main.py in the project root and static/app.js in the project's static/
directory with the supplied files. index.html is included unchanged for reference.

The backend rejects the top prediction when confidence is below 60%. The UI then
shows UNABLE TO IDENTIFY and SENT TO ROOM 101. A confidence of exactly 60% is
accepted.

Restart the FastAPI server after replacing the files. Test with an image whose
highest class score is below 60%, then one at or above 60%.

Note: the model's confidence score is not necessarily a calibrated measure of
real-world correctness. This threshold is a product rule, not proof that the
prediction is objectively correct.
