const express = require('express');
const app = express();
const trainJobworker = require('@tiledesk/tiledesk-train-jobworker')

app.get('/', (req, res) => {
    console.log("Tiledesk Trainer Worker Container works!");
    res.status(200).send("Tiledesk Trainer Worker Container works!");
})

app.listen(3009, () => {
    console.log("Tiledesk Trainer Worker Container listening on port 3009")
})