const express = require('express');
const app = express();

if (process.env.TILELLM_ROLE === 'train') {
    const trainJobworker = require('@tiledesk/tiledesk-train-jobworker')
} else {
    console.log("Worker is on QA module: Train-jobworker disabled!")
}

app.get('/', (req, res) => {
    res.status(200).send("Tiledesk Trainer Worker Container works!");
})

app.listen(3009, () => {
    console.log("Tiledesk Trainer Worker Container listening on port 3009")
})