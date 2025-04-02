# 3. Components & Decomposition

## 1. Components

<style>
    table.custom-table {
        max-width: 1000px;
        width: 100%;
        margin: 0 auto; /* centers the table on the page */
    }

    table.custom-table td {
        background-color: #fff;
    }
    
    table.custom-table th, table.custom-table td {
        text-align: center;
        vertical-align: middle;
        padding: 5px;
        width: 333px; /* distribute the total width equally among three columns */
    }

    table.custom-table img {
        width: 100%;
        display: block; /* removes any gap under the image */
    }
</style>


<table class="custom-table">
    <thead>
        <tr>
            <th>Output: Multiplicative Model</th>
            <th>Output: Additive Model</th>
            <th>Input: Monthly Passengers (In thousands)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                <img src="src/01_output_2.png" alt="Solar Image 1">
            </td>
            <td>
                <img src="src/01_output_1.png" alt="Solar Image 2">
            </td>
            <td>
                <img src="src/01_input.png" alt="Solar Image 3">
            </td>
        </tr>
    </tbody>
</table>

## 2. Additive Decomposition

<style>
    table.custom-table {
        max-width: 1000px;
        width: 100%;
        margin: 0 auto; /* centers the table on the page */
    }

    table.custom-table td {
        background-color: #fff;
    }
    
    table.custom-table th, table.custom-table td {
        text-align: center;
        vertical-align: middle;
        padding: 5px;
        width: 500px; /* distribute the total width equally among three columns */
    }

    table.custom-table img {
        width: 100%;
        display: block; /* removes any gap under the image */
    }
</style>


<table class="custom-table">
    <thead>
        <tr>
            <th>Output: Time Series Decomposition (Additive)</th>
            <th>Input: Original Time Series</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                <img src="src/02_output.png">
            </td>
            <td>
                <img src="src/02_input.png">
            </td>
        </tr>
    </tbody>
</table>

## 3. Multiplicative Decomposition

<style>
    table.custom-table {
        max-width: 600px;
        width: 100%;
        margin: 0 auto; /* centers the table on the page */
    }

    table.custom-table td {
        background-color: #fff;
    }
    
    table.custom-table th, table.custom-table td {
        text-align: center;
        vertical-align: middle;
        padding: 5px;
        width: 600px; /* distribute the total width equally among three columns */
    }

    table.custom-table img {
        width: 100%;
        display: block; /* removes any gap under the image */
    }
</style>


<table class="custom-table">
    <thead>
        <tr>
            <th>Additive vs Multiplicative Model</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                <img src="src/02_output.png">
            </td>
        </tr>
        <tr>
            <td>
                <img src="src/03_output.png">
            </td>
        </tr>
    </tbody>
</table>