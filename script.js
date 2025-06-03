document.addEventListener("DOMContentLoaded", function () {
    const contentRow = document.getElementById('content-row');

    fetch('videos.json')
        .then(response => response.json())
        .then(data => {
            data.videos.forEach(videoData => {
                const pairContainer = document.createElement('div');
                pairContainer.className = 'video-description-pair';

                const videoContainer = document.createElement('div');
                videoContainer.className = 'video-section';

                const videoElement = document.createElement('video');
                videoElement.src = `video/${videoData.filename}`;
                videoElement.controls = true;
                videoElement.preload = 'auto';
                videoElement.style.width = '100%';
                videoElement.style.height = 'auto';

                videoContainer.appendChild(videoElement);

                const descriptionContainer = document.createElement('div');
                descriptionContainer.className = 'description-section';

                const descriptionElement = document.createElement('div');
                descriptionElement.className = 'description';
                descriptionElement.textContent = videoData.description;

                descriptionContainer.appendChild(descriptionElement);

                pairContainer.appendChild(videoContainer);
                pairContainer.appendChild(descriptionContainer);

                contentRow.appendChild(pairContainer);

                videoElement.addEventListener('loadedmetadata', function() {
                    descriptionContainer.style.height = this.clientHeight + 'px';
                });
            });
        })
        .catch(error => console.error('Error fetching the videos:', error));
});
