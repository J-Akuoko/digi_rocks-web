const user_input = document.getElementById("user_input");
const button = document.getElementById("pressed_button");
const display_result = document.getElementById('result')



function increment(value){
    return value += 1
}



function compute_increment(){

    let data = parseFloat(user_input.value);
    let answer = increment(data);
    display_result.textContent = `The increased value is ${answer}`


}


button.onclick = compute_increment;



