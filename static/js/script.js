var submitButton = document.getElementById("submitButton");
var rtmpURL = document.getElementById("rtmpURL");
var labelrtmpURL = document.getElementById("labelrtmpURL");
var image = document.getElementById("image");
var imgButton = document.getElementById("imgButton");

function toggleImage() {
    if (image.style.visibility === "hidden") {
      image.style.visibility = "visible";
      imgButton.style.visibility = "visible"
      image.src="/video";
    } else {
      imgButton.style.visibility = "hidden";
      image.removeAttribute("src");
      image.style.visibility = "hidden";
      submitButton.style.visibility = "visible";
      labelrtmpURL.innerText = "Enter RTMP URL:";
      rtmpURL.style.visibility = "visible";
    }
  }

  function submitURL() {
    if (rtmpURL.checkValidity()){
      submitButton.style.visibility = "hidden";
      labelrtmpURL.innerText = "Displaying the stream from " + rtmpURL.value + ":";
      rtmpURL.style.visibility = "hidden";
      image.style.visibility = "visible";
      imgButton.style.visibility = "visible";
      if (rtmpURL.value === 0){
        image.src=("/video/" + rtmpURL.value);
      }else{
        image.src=("/video/" + encodeURIComponent(rtmpURL.value));
      }
    }
    else{
      alert("Invalid RTMP URL!");
    }

  }

  function cleaning_data() {
    showLoadingOverlay();

    fetch('/clean_data')  // URL to your Flask route
        .then(response => response.json())
        .then(data => {
            // Process your data
            console.log(data.cleaned);
        }) 
        .catch(error => {
            console.error('Error:', error);
        })
        .finally(() => {
            hideLoadingOverlay();
        });
  }

  function hideLoadingOverlay() {
    document.getElementById('loadingOverlay').style.display = 'none';
  }

  function showLoadingOverlay() {
    document.getElementById('loadingOverlay').style.display = 'block';
  }








// // $(".nav li a").on("click", function() {
// //         $(".nav li a").removeClass("active");
// //         $(this).addClass(" active");
// // });

// const currentLocation = location.href;
// console.log('currentlocation = '+currentLocation);
// const menuItem = document.querySelectorAll('a');
// const menuLength = menuItem.length;
// for(let i = 0; i< menuLength; i++){
//     menuItem[i].classList.remove("active");
//     if(menuItem[i].href === currentLocation){
//         console.log('menu item: '+ menuItem[i]);
//         console.log("menu href: " + menuItem[i].href);
//         menuItem[i].className += " active";
//     }
// }


// // set size of container based on sidebar
// // $(window).resize(function() { setSizes(); });

// // function setSizes() {
// //     var sidebarWidth = $("#sidebar").width();
// //     $("#main_content").css({"margin-left":sidebarWidth});

// //     console.log('sidebar width: '+sidebarWidth);
// // }
