const sudokuGrid = document.getElementById('sudoku-grid');
const imageOptions = document.getElementById('image-options');
const message = document.getElementById('message');

// Sample Sudoku puzzle (0 represents empty cells)
const puzzle = [
    [
        9,
        6,
        5,
        2,
        0,
        4,
        3,
        8,
        7
    ],
    [
        3,
        4,
        8,
        5,
        7,
        6,
        1,
        2,
        9
    ],
    [
        2,
        1,
        7,
        3,
        8,
        9,
        4,
        6,
        5
    ],
    [
        1,
        2,
        3,
        0,
        4,
        5,
        0,
        9,
        8
    ],
    [
        8,
        5,
        0,
        1,
        6,
        3,
        0,
        4,
        2
    ],
    [
        6,
        7,
        4,
        9,
        2,
        8,
        0,
        3,
        1
    ],
    [
        0,
        3,
        1,
        4,
        9,
        2,
        8,
        5,
        6
    ],
    [
        5,
        0,
        6,
        8,
        0,
        7,
        2,
        1,
        4
    ],
    [
        4,
        8,
        2,
        6,
        5,
        1,
        0,
        7,
        3
    ]
];

const puzzle_solution = [
    [
        9,
        6,
        5,
        2,
        1,
        4,
        3,
        8,
        7
    ],
    [
        3,
        4,
        8,
        5,
        7,
        6,
        1,
        2,
        9
    ],
    [
        2,
        1,
        7,
        3,
        8,
        9,
        4,
        6,
        5
    ],
    [
        1,
        2,
        3,
        7,
        4,
        5,
        6,
        9,
        8
    ],
    [
        8,
        5,
        9,
        1,
        6,
        3,
        7,
        4,
        2
    ],
    [
        6,
        7,
        4,
        9,
        2,
        8,
        5,
        3,
        1
    ],
    [
        7,
        3,
        1,
        4,
        9,
        2,
        8,
        5,
        6
    ],
    [
        5,
        9,
        6,
        8,
        3,
        7,
        2,
        1,
        4
    ],
    [
        4,
        8,
        2,
        6,
        5,
        1,
        9,
        7,
        3
    ]
];

// Create Sudoku grid
for (let i = 0; i < 9; i++) {
    for (let j = 0; j < 9; j++) {
        const cell = document.createElement('div');
        cell.classList.add('sudoku-cell');
        cell.dataset.row = i;
        cell.dataset.col = j;

        if (puzzle[i][j] !== 0) {
            // Prefill the cell with an image
            const img = document.createElement('img');
            img.src = `./static/images/sudoku/image_${i}_${j}_${puzzle[i][j]}.png`; // Replace with your image paths
            img.classList.add('sudoku-image');
            cell.appendChild(img);
            cell.classList.add('prefilled');
        } else {
            // Empty cell, add event listeners for drag and drop
            cell.addEventListener('dragover', dragOver);
            cell.addEventListener('drop', drop);
        }

        sudokuGrid.appendChild(cell);
    }
}

// Create image options
for (let i = 1; i <= 9; i++) {
    const img = document.createElement('img');
    img.src = `./static/images/sudoku/image_${i}.png`; // Replace with your image paths
    img.classList.add('image-option');
    img.draggable = true;
    img.addEventListener('dragstart', dragStart);
    imageOptions.appendChild(img);
}

let draggedImage = null;

function dragStart(e) {
    draggedImage = e.target;
}

function dragOver(e) {
    e.preventDefault();
}

function drop(e) {
    e.preventDefault();
    if (e.target.classList.contains('sudoku-cell') && !e.target.classList.contains('prefilled')) {
        e.target.innerHTML = '';
        e.target.appendChild(draggedImage.cloneNode());
        checkSolution();
    }
}

function checkSolution() {
    let solved = true;
    let wrong = false;
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const cell = sudokuGrid.children[i * 9 + j];
            const img = cell.querySelector('img');

            // if img is null, set solved to false
            if (!img) {
                solved = false;
                continue;
            }
            // if img is false, set wrong to true
            if (parseInt(img.src.slice(-5, -4)) !== puzzle_solution[i][j]) {
                wrong = true;
                solved = false;
                break;
            }
        }
    }

    if (solved) {
        showModal('Congratulations, you solved the Sudoku! Can you build an AI that can solve it too?');
    }
    if (wrong) {
        showSimpleModal('You made a mistake. Keep trying! (Please refresh)');
    }
}


function showModal(message) {
    const modal = document.getElementById('modal');
    const modalMessage = document.getElementById('modal-message');
    const closeBtn = document.getElementsByClassName('close')[0];
    const canvas = document.getElementById('fireworks-canvas');

    modalMessage.textContent = message;
    modal.style.display = 'block';

    // Set canvas size
    canvas.width = modal.clientWidth;
    canvas.height = modal.clientHeight;

    // Start fireworks
    fireworksDisplay = new FireworksDisplay(canvas);
    fireworksDisplay.start();

    closeBtn.onclick = function () {
        modal.style.display = 'none';
        fireworksDisplay.stop();
    }

    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = 'none';
            fireworksDisplay.stop();
        }
    }
}

function showSimpleModal(message) {
    const modal = document.getElementById('modal');
    const modalMessage = document.getElementById('modal-message');
    const closeBtn = document.getElementsByClassName('close')[0];

    modalMessage.textContent = message;
    modal.style.display = 'block';

    closeBtn.onclick = function () {
        modal.style.display = 'none';
    }

    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }
}
