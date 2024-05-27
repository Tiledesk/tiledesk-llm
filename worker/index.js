const express = require('express');
const app = express();

if (process.env.TILELLM_ROLE === 'train') {
    const trainJobworker = require('@tiledesk/tiledesk-train-jobworker')
} else {
    console.log("Worker non parte!!!!!!!")
}

app.get('/', (req, res) => {
    console.log("Tiledesk Trainer Worker Container works!");
    if (trainJobworker) {
        console.log("(Test Log) trainJobworker: ", trainJobworker);
    }
    res.status(200).send("Tiledesk Trainer Worker Container works!");
})

app.listen(3009, () => {
    console.log("Tiledesk Trainer Worker Container listening on port 3009")
})