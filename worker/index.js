const express = require('express');
const app = express();

app.get('/', (req, res) => {
    console.log("Tiledesk Trainer Worker works!");
    res.status(200).send("Tiledesk Trainer Worker works!");
})

app.listen(3009, () => {
    console.log("app listening on port 3009")
})