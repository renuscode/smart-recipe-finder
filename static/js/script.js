const uploadButton =
    document.getElementById("uploadButton");

const cameraButton =
    document.getElementById("cameraButton");

const foodImage =
    document.getElementById("foodImage");

const preview =
    document.getElementById("preview");

const emptyPreview =
    document.getElementById("emptyPreview");

const analysisTitle =
    document.getElementById("analysisTitle");

const analysisText =
    document.getElementById("analysisText");


/* Upload button */

if (uploadButton) {

    uploadButton.addEventListener("click", function () {

        foodImage.click();

    });

}


/* Image selection */

if (foodImage) {

    foodImage.addEventListener("change", function (event) {

        const file = event.target.files[0];

        if (!file) {
            return;
        }

        if (!file.type.startsWith("image/")) {

            alert("Please select a valid food image.");

            return;
        }


        const imageURL =
            URL.createObjectURL(file);

        preview.src = imageURL;

        preview.style.display = "block";

        emptyPreview.style.display = "none";


        analysisTitle.innerText =
            "Image ready for analysis";

        analysisText.innerText =
            "The image is ready to be processed.";

    });

}


/* Camera */

if (cameraButton) {

    cameraButton.addEventListener("click", function () {

        alert(
            "Camera functionality will be connected to the backend."
        );

    });

}


/* Select dish */

function selectDish(dishName) {

    const selectedDish =
        document.getElementById("selectedDish");

    if (selectedDish) {

        selectedDish.innerText =
            dishName;

    }

    localStorage.setItem(
        "selectedDish",
        dishName
    );

}


/* Go to recipe */

function goToRecipe() {

    const selectedDish =
        document.getElementById("selectedDish");

    if (selectedDish) {

        localStorage.setItem(
            "selectedDish",
            selectedDish.innerText
        );

    }

    window.location.href = "recipe.html";
}

    


/* Try another image */

function tryAnotherImage() {

    if (foodImage) {

        foodImage.value = "";

        foodImage.click();

    }

}


/* Load selected dish on recipe page */

window.addEventListener("DOMContentLoaded", function () {

    const savedDish =
        localStorage.getItem("selectedDish");

    const recipeDishName =
        document.getElementById("recipeDishName");

    if (savedDish && recipeDishName) {

        recipeDishName.innerText =
            savedDish;

    }

});