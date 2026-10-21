/* =========================================================
   Smart Recipe Finder
   Frontend Controller
   jQuery + Bootstrap 5
========================================================= */

$(document).ready(function () {

    let videoStream = null;
    let currentClassification = null;

    // =========================================================
    // ELEMENTS
    // =========================================================

    const $foodImageInput = $("#foodImageInput");
    const $chooseFileBtn = $("#chooseFileBtn");
    const $uploadModeBtn = $("#uploadModeBtn");
    const $cameraModeBtn = $("#cameraModeBtn");
    const $openCameraBtn = $("#openCameraBtn");
    const $takeSnapshotBtn = $("#takeSnapshotBtn");

    const cameraModalElement = document.getElementById("cameraModal");
    const cameraModal = cameraModalElement
        ? bootstrap.Modal.getOrCreateInstance(cameraModalElement)
        : null;

    const cameraVideo = document.getElementById("cameraVideo");
    const cameraCanvas = document.getElementById("cameraCanvas");

    const $activeImagePreview = $("#activeImagePreview");
    const $emptyPreviewBox = $("#emptyPreviewBox");
    const $scanLaserLine = $("#scanLaserLine");

    const $statusArea = $("#statusArea");
    const $statusTitle = $("#statusTitle");
    const $statusSubtitle = $("#statusSubtitle");
    const $statusSpinner = $("#statusSpinner");

    const $resultsSection = $("#resultsSection");
    const $topDishesList = $("#topDishesList");

    const $bestDishTitle = $("#bestDishTitle");
    const $bestVarietySubtitle = $("#bestVarietySubtitle");
    const $bestDishDesc = $("#bestDishDesc");

    const $viewRecipeBtn = $("#viewRecipeBtn");
    const $tryAnotherBtn = $("#tryAnotherBtn");


    // =========================================================
    // HELPER: SHOW STATUS
    // =========================================================

    function showAnalyzingStatus() {

        $statusArea.removeClass("d-none");
        $statusSpinner.removeClass("d-none");

        $statusTitle.text("Analyzing your food...");

        $statusSubtitle.text(
            "Our local AI is identifying the dish and matching the recipe."
        );

        $scanLaserLine.removeClass("d-none");

        $resultsSection.addClass("d-none");
    }


    // =========================================================
    // HELPER: SHOW ERROR
    // =========================================================

    function showClassificationError(message) {

        $scanLaserLine.addClass("d-none");
        $statusSpinner.addClass("d-none");

        $statusArea.removeClass("d-none");

        $statusTitle.text("Recognition Failed");

        $statusSubtitle.text(
            message || "Unable to identify the food image."
        );
    }


    // =========================================================
    // FILE BUTTON
    // =========================================================

    if ($chooseFileBtn.length) {

        $chooseFileBtn.on("click", function () {

            $foodImageInput.trigger("click");

        });
    }


    // =========================================================
    // UPLOAD MODE
    // =========================================================

    if ($uploadModeBtn.length) {

        $uploadModeBtn.on("click", function () {

            $uploadModeBtn.css({
                background: "linear-gradient(135deg, #45a95f, #287d43)",
                color: "white",
                border: "none"
            });

            $cameraModeBtn.css({
                background: "white",
                color: "#287d43",
                border: "2px solid #287d43"
            });

            $foodImageInput.trigger("click");

        });
    }


    // =========================================================
    // IMAGE SELECTED
    // =========================================================

    $foodImageInput.on("change", function (event) {

        const file = event.target.files[0];

        if (!file) {
            return;
        }

        console.log("[UPLOAD] Selected file:", file.name);
        console.log("[UPLOAD] File type:", file.type);
        console.log("[UPLOAD] File size:", file.size);

        // Validate image
        if (!file.type.startsWith("image/")) {

            alert("Please select a valid image file.");

            $foodImageInput.val("");

            return;
        }

        // Preview
        const objectUrl = URL.createObjectURL(file);

        displayPreviewImage(objectUrl);

        // Show loading
        showAnalyzingStatus();

        // Create FormData
        const formData = new FormData();

        // IMPORTANT:
        // Flask expects request.files["food_image"]
        formData.append("food_image", file);

        console.log("[UPLOAD] Sending image to /api/classify...");

        sendClassificationRequest(formData, false);
    });


    // =========================================================
    // DISPLAY PREVIEW
    // =========================================================

    function displayPreviewImage(srcUrl) {

        $emptyPreviewBox.addClass("d-none");

        $activeImagePreview
            .attr("src", srcUrl)
            .removeClass("d-none");

        $scanLaserLine.removeClass("d-none");

        $statusArea.removeClass("d-none");

        $resultsSection.addClass("d-none");
    }


    // =========================================================
    // CAMERA
    // =========================================================

    function openCamera() {

        if (!cameraModal) {

            alert("Camera modal could not be initialized.");

            return;
        }

        cameraModal.show();

        if (
            navigator.mediaDevices &&
            navigator.mediaDevices.getUserMedia
        ) {

            navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: {
                        ideal: "environment"
                    },

                    width: {
                        ideal: 1280
                    },

                    height: {
                        ideal: 720
                    }
                }

            })

            .then(function (stream) {

                videoStream = stream;

                cameraVideo.srcObject = stream;

                cameraVideo.play();

                console.log("[CAMERA] Camera started.");

            })

            .catch(function (error) {

                console.error(
                    "[CAMERA] Camera access error:",
                    error
                );

                alert(
                    "Could not access camera. Please allow camera permission."
                );

                cameraModal.hide();
            });

        } else {

            alert(
                "Camera capture is not supported by this browser."
            );

            cameraModal.hide();
        }
    }


    // =========================================================
    // STOP CAMERA
    // =========================================================

    function stopCamera() {

        if (videoStream) {

            videoStream
                .getTracks()
                .forEach(function (track) {

                    track.stop();

                });

            videoStream = null;
        }

        if (cameraVideo) {

            cameraVideo.srcObject = null;
        }

        console.log("[CAMERA] Camera stopped.");
    }


    // =========================================================
    // OPEN CAMERA BUTTON
    // =========================================================

    if ($openCameraBtn.length) {

        $openCameraBtn.on("click", function () {

            openCamera();

        });
    }


    // =========================================================
    // CAMERA MODE BUTTON
    // =========================================================

    if ($cameraModeBtn.length) {

        $cameraModeBtn.on("click", function () {

            $cameraModeBtn.css({
                background: "linear-gradient(135deg, #45a95f, #287d43)",
                color: "white",
                border: "none"
            });

            $uploadModeBtn.css({
                background: "white",
                color: "#287d43",
                border: "2px solid #287d43"
            });

            openCamera();

        });
    }


    // =========================================================
    // CAMERA MODAL CLOSED
    // =========================================================

    if (cameraModalElement) {

        cameraModalElement.addEventListener(
            "hidden.bs.modal",
            function () {

                stopCamera();

            }
        );
    }


    // =========================================================
    // CAPTURE CAMERA IMAGE
    // =========================================================

    if ($takeSnapshotBtn.length) {

        $takeSnapshotBtn.on("click", function () {

            if (
                !cameraVideo ||
                !cameraVideo.videoWidth ||
                !cameraVideo.videoHeight
            ) {

                alert(
                    "Camera is not ready yet. Please wait a moment."
                );

                return;
            }

            // Set canvas size
            cameraCanvas.width = cameraVideo.videoWidth;

            cameraCanvas.height = cameraVideo.videoHeight;

            const context = cameraCanvas.getContext("2d");

            context.drawImage(
                cameraVideo,
                0,
                0,
                cameraCanvas.width,
                cameraCanvas.height
            );

            // Convert to JPEG
            const dataUrl =
                cameraCanvas.toDataURL(
                    "image/jpeg",
                    0.92
                );

            console.log("[CAMERA] Snapshot captured.");

            // Stop camera
            stopCamera();

            // Close modal
            if (cameraModal) {

                cameraModal.hide();
            }

            // Preview captured image
            displayPreviewImage(dataUrl);

            // Show analyzing
            showAnalyzingStatus();

            console.log(
                "[CAMERA] Sending image to /api/classify..."
            );

            // IMPORTANT:
            // Flask expects JSON:
            // {"camera_image": "..."}
            sendClassificationRequest(
                {
                    camera_image: dataUrl
                },
                true
            );
        });
    }


    // =========================================================
    // CLASSIFICATION REQUEST
    // =========================================================

    function sendClassificationRequest(payload, isJson) {

        console.log(
            "[API] POST /api/classify"
        );

        $.ajax({

            url: "/api/classify",

            type: "POST",

            data: isJson
                ? JSON.stringify(payload)
                : payload,

            contentType: isJson
                ? "application/json"
                : false,

            processData: isJson
                ? false
                : false,

            cache: false,

            success: function (response) {

                console.log(
                    "[API] Classification response:",
                    response
                );

                $scanLaserLine.addClass("d-none");

                $statusSpinner.addClass("d-none");


                // -------------------------------------------------
                // SUCCESS
                // -------------------------------------------------

                if (response && response.success) {

                    $statusTitle.text(
                        "Dish Identified Successfully!"
                    );

                    $statusSubtitle.text(
                        "Recipe found successfully."
                    );

                    renderRecognitionResults(response);

                    return;
                }


                // -------------------------------------------------
                // BACKEND RETURNED ERROR
                // -------------------------------------------------

                const errorMessage =
                    response && response.error
                        ? response.error
                        : "Unable to identify the dish.";

                showClassificationError(
                    errorMessage
                );
            },

            error: function (xhr, status, error) {

                console.error(
                    "[API] Classification AJAX error"
                );

                console.error(
                    "Status:",
                    status
                );

                console.error(
                    "Error:",
                    error
                );

                console.error(
                    "HTTP status:",
                    xhr.status
                );

                console.error(
                    "Response:",
                    xhr.responseText
                );

                $scanLaserLine.addClass("d-none");

                $statusSpinner.addClass("d-none");


                let errorMessage =
                    "Server error while recognizing food.";


                if (xhr.responseJSON) {

                    if (xhr.responseJSON.error) {

                        errorMessage =
                            xhr.responseJSON.error;
                    }
                }


                showClassificationError(
                    errorMessage
                );

                alert(
                    "Recognition Error:\n\n" +
                    errorMessage
                );
            }
        });
    }


    // =========================================================
    // RENDER RECOGNITION RESULTS
    // =========================================================

    function renderRecognitionResults(data) {

        currentClassification = data;

        $topDishesList.empty();


        // -------------------------------------------------------
        // TOP DISHES
        // -------------------------------------------------------

        if (
            data.top_dishes &&
            Array.isArray(data.top_dishes) &&
            data.top_dishes.length > 0
        ) {

            data.top_dishes.forEach(
                function (dish, index) {

                    const isActive =
                        index === 0
                            ? "active"
                            : "";

                    const match =
                        Number(
                            dish.match_percent || 0
                        );


                    const dishCardHtml = `

                        <div
                            class="dish-match-card ${isActive}"
                            data-dish="${escapeHtml(dish.name || "")}"
                            data-raw="${escapeHtml(dish.raw_name || "")}"
                        >

                            <div class="dish-match-header">

                                <span class="dish-match-title">

                                    <i class="fa-solid fa-utensils text-success me-2"></i>

                                    ${escapeHtml(dish.name || "Unknown Dish")}

                                </span>

                                <span class="dish-match-percent">

                                    ${match}% Match

                                </span>

                            </div>


                            <div class="dish-match-progress">

                                <div
                                    class="dish-match-progress-bar"
                                    style="width:${Math.min(
                                        100,
                                        Math.max(0, match)
                                    )}%"
                                ></div>

                            </div>

                        </div>
                    `;

                    $topDishesList.append(
                        dishCardHtml
                    );
                }
            );

        } else {

            $topDishesList.html(
                `
                <div class="alert alert-light">
                    No top dish predictions available.
                </div>
                `
            );
        }


        // -------------------------------------------------------
        // BEST MATCH
        // -------------------------------------------------------

        updateBestMatchPanel(

            data.best_dish || "Unknown Dish",

            data.predicted_variety || "",

            data.confidence_percent || 0,

            data.user_image_url || ""
        );


        // -------------------------------------------------------
        // DESCRIPTION
        // -------------------------------------------------------

        if (data.description) {

            $bestDishDesc.text(
                data.description
            );

        } else {

            $bestDishDesc.text(
                "A recipe was found in our dataset with complete ingredients and cooking instructions."
            );
        }


        // -------------------------------------------------------
        // SHOW RESULTS
        // -------------------------------------------------------

        $resultsSection.removeClass(
            "d-none"
        );


        // Scroll
        if ($resultsSection.offset()) {

            $("html, body").animate(

                {
                    scrollTop:
                        $resultsSection.offset().top - 80
                },

                600
            );
        }
    }


    // =========================================================
    // UPDATE BEST MATCH
    // =========================================================

    function updateBestMatchPanel(
        dishName,
        varietyName,
        confidence,
        userImageUrl
    ) {

        $bestDishTitle.text(
            dishName || "Unknown Dish"
        );


        if (varietyName) {

            const formattedVariety =
                String(varietyName)
                    .replace(
                        /\b\w/g,
                        function (letter) {
                            return letter.toUpperCase();
                        }
                    );


            $bestVarietySubtitle.html(

                `Predicted Variety: <strong>${escapeHtml(
                    formattedVariety
                )}</strong> (${confidence}% confidence)`

            );

        } else {

            $bestVarietySubtitle.text(
                `Identified Dish: ${dishName}`
            );
        }


        // Top predictions
        const topDishes =
            currentClassification &&
            currentClassification.top_dishes
                ? currentClassification.top_dishes
                : [];


        const topPredictionsJson =
            JSON.stringify(topDishes);


        // Recipe URL
        const recipeUrl =
            `/recipe?dish=${encodeURIComponent(
                dishName || ""
            )}` +
            `&variety=${encodeURIComponent(
                varietyName || ""
            )}` +
            `&user_image=${encodeURIComponent(
                userImageUrl || ""
            )}` +
            `&top_predictions=${encodeURIComponent(
                topPredictionsJson
            )}`;


        $viewRecipeBtn.attr(
            "href",
            recipeUrl
        );


        console.log(
            "[RECIPE] View Recipe URL:",
            recipeUrl
        );
    }


    // =========================================================
    // TOP DISH CLICK
    // =========================================================
    //
    // IMPORTANT:
    // The ML predicted dish must remain the recipe selection.
    //
    // The other prediction cards are alternatives for display.
    // Clicking them must NOT change:
    //
    // 1. Best Match
    // 2. Predicted Variety
    // 3. View Recipe URL
    //
    // Example:
    //
    // Puran Poli       59.2%
    // Aloo Paratha     31.8%
    // Chapati           5.8%
    //
    // If the user clicks Aloo Paratha, only the card is
    // highlighted. The recipe remains Puran Poli.
    // =========================================================

    $(document).on(
    "click",
    ".dish-match-card",
    function () {

        $(".dish-match-card")
            .removeClass("active");

        $(this).addClass("active");

        const selectedDish =
            $(this).attr("data-dish") || "";

        console.log(
            "[TOP PREDICTION] Selected recipe:",
            selectedDish
        );

        const userImg =
            currentClassification
                ? currentClassification.user_image_url
                : "";

        // Update View Recipe button
        // according to the selected prediction.
        updateBestMatchPanel(
            selectedDish,
            "",
            100,
            userImg
        );
    }
);


    // =========================================================
    // TRY ANOTHER IMAGE
    // =========================================================

    if ($tryAnotherBtn.length) {

        $tryAnotherBtn.on(
            "click",
            function () {

                $foodImageInput.val("");

                $resultsSection.addClass(
                    "d-none"
                );

                $activeImagePreview
                    .attr("src", "")
                    .addClass("d-none");

                $emptyPreviewBox
                    .removeClass("d-none");

                $statusArea
                    .addClass("d-none");

                $scanLaserLine
                    .addClass("d-none");

                $foodImageInput.trigger(
                    "click"
                );
            }
        );
    }


    // =========================================================
    // BASIC HTML ESCAPE
    // =========================================================

    function escapeHtml(value) {

        return String(value || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }


    // =========================================================
    // INITIAL LOG
    // =========================================================

    console.log(
        "Smart Recipe Finder frontend loaded successfully."
    );

});